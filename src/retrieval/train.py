from config import load_config
from data.transforms import build_retrieval_train_transform
from utils import cleanup_distributed, reduce_mean, save_json, set_seed, setup_distributed, unwrap_model
from .datasets import RetrievalPairDataset, require_retrieval_manifests
from .model import DINOv2GeMEncoder, SimSiamRetrievalModel, simsiam_loss
from .utils import _retrieval_output_dir

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def train_retrieval(
	args,
):
    config = load_config(args.profile)
    set_seed(int(config.data["runtime"]["seed"]))
    dist_state, device = setup_distributed(args.device)
    try:
        base_rows, _ = require_retrieval_manifests(config)
        batch_size = int(config.data["retrieval"]["batch_size"])
        use_pretrained = not bool(getattr(args, "no_pretrained", False))
        encoder = DINOv2GeMEncoder(config.data["retrieval"]["backbone_name"], pretrained=use_pretrained)
        train_adapter = bool(config.data["retrieval"].get("train_adapter", True))
        if not train_adapter:
            if dist_state.is_main:
                out_dir = _retrieval_output_dir(config)
                checkpoint = out_dir / "model.pt"
                torch.save(
                    {
                        "model_name": config.data["retrieval"]["backbone_name"],
                        "pretrained": use_pretrained,
                        "train_adapter": False,
                        "encoder": encoder.state_dict(),
                    },
                    checkpoint,
                )
                save_json(
                    out_dir / "train_summary.json",
                    {
                        "losses": [],
                        "device": str(device),
                        "distributed": dist_state.enabled,
                        "world_size": dist_state.world_size,
                        "pretrained": use_pretrained,
                        "train_adapter": False,
                    },
                )
                print(f"Saved frozen retrieval checkpoint to {checkpoint}")
            return
        transform = build_retrieval_train_transform(encoder.input_size)
        dataset = RetrievalPairDataset(base_rows, transform)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if dist_state.enabled else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(config.data["runtime"]["num_workers"]),
            drop_last=True,
            pin_memory=device.type == "cuda",
        )
        model = SimSiamRetrievalModel(encoder).to(device)
        if dist_state.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
            )
        lr = float(config.data["retrieval"]["learning_rate"])
        wd = float(config.data["retrieval"]["weight_decay"])
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=wd)
        epochs = int(args.epochs or config.data["retrieval"]["epochs"])
        max_batches = args.max_batches

        model.train()
        losses: list[float] = []
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"retrieval epoch {epoch + 1}/{epochs}",
                disable=not dist_state.is_main,
            )
            running_loss = 0.0
            steps_run = 0
            for step, (view1, view2) in enumerate(progress, start=1):
                view1 = view1.to(device, non_blocking=True)
                view2 = view2.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                p1, p2, z1, z2 = model(view1, view2)
                loss = 0.5 * (simsiam_loss(p1, z2) + simsiam_loss(p2, z1))
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
                steps_run = step
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if max_batches is not None and step >= max_batches:
                    break
            epoch_loss = running_loss / max(1, steps_run)
            losses.append(reduce_mean(epoch_loss, device, dist_state))

        if dist_state.is_main:
            out_dir = _retrieval_output_dir(config)
            checkpoint = out_dir / "model.pt"
            trained_model = unwrap_model(model)
            torch.save(
                {
                    "model_name": config.data["retrieval"]["backbone_name"],
                    "pretrained": use_pretrained,
                    "train_adapter": True,
                    "encoder": trained_model.encoder.state_dict(),
                    "projector": trained_model.projector.state_dict(),
                    "predictor": trained_model.predictor.state_dict(),
                },
                checkpoint,
            )
            save_json(
                out_dir / "train_summary.json",
                {
                    "losses": losses,
                    "device": str(device),
                    "distributed": dist_state.enabled,
                    "world_size": dist_state.world_size,
                    "train_adapter": True,
                },
            )
            print(f"Saved retrieval checkpoint to {checkpoint}")
    finally:
        cleanup_distributed(dist_state)
