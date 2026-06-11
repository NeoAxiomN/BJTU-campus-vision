from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import torch
from torch import nn

DETECTOR_MODEL_TYPE = "fasterrcnn_mobilenet_v3_large_320_fpn"
DETECTOR_NUM_CLASSES = 2


def build_detector(
	num_classes: int = DETECTOR_NUM_CLASSES,
	pretrained: bool = True,
) -> nn.Module:
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    try:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights,
            weights_backbone=None if not pretrained else None,
        )
    except Exception as exc:  # pragma: no cover - depends on external weight cache/network
        raise RuntimeError(
            "Failed to initialize Faster R-CNN MobileNetV3 weights. Ensure network access "
            "to download.pytorch.org or place the torchvision checkpoint in the torch cache."
        ) from exc
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def move_targets_to_device(
	targets: list[dict[str, torch.Tensor]],
	device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    moved: list[dict[str, torch.Tensor]] = []
    for target in targets:
        moved.append({key: value.to(device) for key, value in target.items()})
    return moved
