from config import load_config
from data.transforms import build_detection_image_transform
from utils import cleanup_distributed, reduce_mean, save_json, set_seed, setup_distributed, unwrap_model
from .datasets import DetectionDataset, require_detection_manifests
from .model import DETECTOR_MODEL_TYPE, DETECTOR_NUM_CLASSES, build_detector, move_targets_to_device
from .utils import _collate, _detection_output_dir, _move_images_to_device

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def train_detector(
	args,
):
    config = load_config(args.profile)
    set_seed(int(config.data["runtime"]["seed"]))
    dist_state, device = setup_distributed(args.device)
    try:
        train_rows, _ = require_detection_manifests(config)
        dataset = DetectionDataset(train_rows, build_detection_image_transform(int(config.data["detector"]["image_size"])))
        sampler = DistributedSampler(dataset, shuffle=True) if dist_state.enabled else None
        loader = DataLoader(
            dataset,
            batch_size=int(config.data["detector"]["batch_size"]),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(config.data["runtime"]["num_workers"]),
            collate_fn=_collate,
            pin_memory=device.type == "cuda",
        )
        use_pretrained = not bool(getattr(args, "no_pretrained", False))
        model = build_detector(pretrained=use_pretrained).to(device)
        if dist_state.enabled:
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                broadcast_buffers=False,
            )
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=float(config.data["detector"]["learning_rate"]),
            momentum=float(config.data["detector"].get("momentum", 0.9)),
            weight_decay=float(config.data["detector"].get("weight_decay", 0.0005)),
        )
        epochs = int(args.epochs or config.data["detector"]["epochs"])
        max_batches = args.max_batches
        loss_trace: list[float] = []

        model.train()
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"detector epoch {epoch + 1}/{epochs}",
                disable=not dist_state.is_main,
            )
            running = 0.0
            steps_run = 0
            for step, batch in enumerate(progress, start=1):
                images = _move_images_to_device(batch["image"], device)
                targets = move_targets_to_device(batch["target"], device)
                optimizer.zero_grad(set_to_none=True)
                losses = model(images, targets)
                loss = sum(loss_value for loss_value in losses.values())
                loss.backward()
                optimizer.step()
                running += float(loss.item())
                steps_run = step
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if max_batches is not None and step >= max_batches:
                    break
            epoch_loss = running / max(1, steps_run)
            loss_trace.append(reduce_mean(epoch_loss, device, dist_state))

        if dist_state.is_main:
            out_dir = _detection_output_dir(config)
            checkpoint = out_dir / "model.pt"
            torch.save(
                {
                    "model_type": DETECTOR_MODEL_TYPE,
                    "num_classes": DETECTOR_NUM_CLASSES,
                    "pretrained": use_pretrained,
                    "model": unwrap_model(model).state_dict(),
                },
                checkpoint,
            )
            save_json(
                out_dir / "train_summary.json",
                {
                    "losses": loss_trace,
                    "device": str(device),
                    "distributed": dist_state.enabled,
                    "world_size": dist_state.world_size,
                    "model_type": DETECTOR_MODEL_TYPE,
                    "pretrained": use_pretrained,
                },
            )
            print(f"Saved detection checkpoint to {checkpoint}")
    finally:
        cleanup_distributed(dist_state)
