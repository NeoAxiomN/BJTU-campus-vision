from utils import ensure_dir
from .model import DETECTOR_MODEL_TYPE, build_detector

from pathlib import Path

import torch


def _detection_output_dir(
	config,
) -> Path:
    return ensure_dir(config.paths.output_root / config.profile / "detection")


def _load_model(
	config,
	device: torch.device,
):
    checkpoint_path = _detection_output_dir(config) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Detection checkpoint is missing. Run `train-detector --profile "
            f"{config.profile}` first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_type") != DETECTOR_MODEL_TYPE:
        raise RuntimeError(
            "Detection checkpoint is incompatible with the Faster R-CNN detector. "
            "Rerun `train-detector` to create a new checkpoint."
        )
    model = build_detector(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def _score_threshold(
	config,
) -> float:
    return float(config.data["detector"].get("score_threshold", 0.35))


def _collate(
	batch: list[dict],
) -> dict:
    return {
        "image": [item["image"] for item in batch],
        "target": [item["target"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
        "id": [item["id"] for item in batch],
        "label": [item["label"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def _move_images_to_device(
	images: list[torch.Tensor],
	device: torch.device,
) -> list[torch.Tensor]:
    return [image.to(device, non_blocking=True) for image in images]


def _clamp_boxes(
	boxes: list[list[float]],
	width: int,
	height: int,
) -> list[list[float]]:
    clamped: list[list[float]] = []
    for x1, y1, x2, y2 in boxes:
        cx1 = max(0.0, min(float(width), float(x1)))
        cy1 = max(0.0, min(float(height), float(y1)))
        cx2 = max(0.0, min(float(width), float(x2)))
        cy2 = max(0.0, min(float(height), float(y2)))
        if cx2 > cx1 and cy2 > cy1:
            clamped.append([cx1, cy1, cx2, cy2])
    return clamped
