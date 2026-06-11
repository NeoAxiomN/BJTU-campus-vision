from data.manifests import det_train_manifest_path, det_val_manifest_path
from utils import load_jsonl
from .utils import _clamp_boxes

import torch
from PIL import Image
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(
    	self,
    	rows: list[dict],
    	transform,
    ):
        self.rows = rows
        self.transform = transform

    def __len__(
    	self,
    ) -> int:
        return len(self.rows)

    def __getitem__(
    	self,
    	index: int,
    ):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        clamped_boxes = _clamp_boxes(row["boxes"], row["width"], row["height"])
        boxes = torch.tensor(clamped_boxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        sample = {
            "image": image_tensor,
            "target": target,
            "boxes": boxes,
            "id": row["id"],
            "label": row["label"],
            "image_path": row["image_path"],
        }
        return sample


def load_detection_rows(
	config,
) -> tuple[list[dict], list[dict]]:
    train_rows = load_jsonl(det_train_manifest_path(config))
    val_rows = load_jsonl(det_val_manifest_path(config))
    return train_rows, val_rows


def require_detection_manifests(
	config,
) -> tuple[list[dict], list[dict]]:
    try:
        return load_detection_rows(config)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Detection manifests are missing. Run `prepare-data --profile "
            f"{config.profile}` first."
        ) from exc
