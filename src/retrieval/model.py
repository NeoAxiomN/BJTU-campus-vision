import torch
import torch.nn.functional as F
from torch import nn

try:
    import timm
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
    timm = None
    TIMM_IMPORT_ERROR = exc
else:
    TIMM_IMPORT_ERROR = None


class GeM(nn.Module):
    def __init__(
    	self,
    	p: float = 3.0,
    	eps: float = 1e-6,
    ):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(
    	self,
    	x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)
        return x.pow(1.0 / self.p)


class DINOv2GeMEncoder(nn.Module):
    def __init__(
    	self,
    	model_name: str,
    	pretrained: bool = True,
    ):
        super().__init__()
        if TIMM_IMPORT_ERROR is not None:
            raise RuntimeError("timm is required for the retrieval model.") from TIMM_IMPORT_ERROR
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as exc:  # pragma: no cover - depends on external weights
            raise RuntimeError(
                "Failed to initialize DINOv2 weights. Ensure network access or a local timm cache."
            ) from exc
        self.pool = GeM()
        self.embedding_dim = getattr(self.backbone, "num_features", 768)
        self._freeze_parameters()

    @property
    def input_size(
    	self,
    ) -> int:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:
            return 224
        image_size = getattr(patch_embed, "img_size", 224)
        if isinstance(image_size, tuple):
            return int(image_size[0])
        return int(image_size)

    def _freeze_parameters(
    	self,
    ):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is not None:
            for parameter in patch_embed.parameters():
                parameter.requires_grad = False
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            return
        trainable = list(blocks)[-2:]
        for block in trainable:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def _extract_tokens(
    	self,
    	x: torch.Tensor,
    ) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                return features["x_norm_patchtokens"]
            if "x_prenorm" in features:
                prenorm = features["x_prenorm"]
                return prenorm[:, 1:] if prenorm.dim() == 3 and prenorm.size(1) > 1 else prenorm
            if "x" in features:
                features = features["x"]
        if isinstance(features, (list, tuple)):
            features = features[0]
        if features.dim() == 2:
            return features.unsqueeze(1)
        if features.size(1) > 1:
            return features[:, 1:]
        return features

    def forward(
    	self,
    	x: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._extract_tokens(x)
        pooled = self.pool(tokens)
        return F.normalize(pooled, dim=-1)


class SimSiamRetrievalModel(nn.Module):
    def __init__(
    	self,
    	encoder: DINOv2GeMEncoder,
    ):
        super().__init__()
        dim = encoder.embedding_dim
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
        )

    def forward(
    	self,
    	view1: torch.Tensor,
    	view2: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        z1 = self.projector(self.encoder(view1))
        z2 = self.projector(self.encoder(view2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()


def simsiam_loss(
	p: torch.Tensor,
	z: torch.Tensor,
) -> torch.Tensor:
    return -F.cosine_similarity(F.normalize(p, dim=1), F.normalize(z, dim=1), dim=1).mean()
