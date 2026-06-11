from config import load_config
from data.transforms import build_detection_image_transform
from utils import ensure_dir, save_json
from utils.device import resolve_device
from utils.visualization import draw_boxes, save_image_grid
from .datasets import DetectionDataset, require_detection_manifests
from .postprocess import detection_prf
from .utils import _collate, _detection_output_dir, _load_model, _score_threshold

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


def filter_predictions(
	output: dict[str, torch.Tensor],
	score_threshold: float,
) -> tuple[list[list[float]], list[float]]:
    boxes = output["boxes"].detach().cpu()
    scores = output["scores"].detach().cpu()
    keep = scores >= score_threshold
    kept_boxes = boxes[keep].tolist()
    kept_scores = scores[keep].tolist()
    return kept_boxes, kept_scores


def evaluate_detector(
	args,
):
    config = load_config(args.profile)
    device = resolve_device(args.device)
    _, val_rows = require_detection_manifests(config)
    dataset = DetectionDataset(val_rows, build_detection_image_transform(int(config.data["detector"]["image_size"])))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=_collate)
    model = _load_model(config, device)
    threshold = _score_threshold(config)
    row_lookup = {row["id"]: row for row in val_rows}

    metrics = defaultdict(list)
    prediction_counts: list[int] = []
    score_values: list[float] = []
    vis_dir = ensure_dir(_detection_output_dir(config) / "visualizations")
    with torch.no_grad():
        for batch in loader:
            images = [image.to(device) for image in batch["image"]]
            output = model(images)[0]
            pred_boxes, pred_scores = filter_predictions(output, threshold)
            gt_boxes = batch["boxes"][0].tolist()
            precision, recall, f1 = detection_prf(pred_boxes, gt_boxes)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            prediction_counts.append(len(pred_boxes))
            score_values.extend(pred_scores)
            image_path = Path(batch["image_path"][0])
            row = row_lookup[batch["id"][0]]
            image = draw_boxes(image_path, row["boxes"], color="lime")
            pred_image = draw_boxes(image_path, pred_boxes, color="yellow")
            save_image_grid(
                vis_dir / f"{batch['id'][0]}.png",
                [image, pred_image],
                ["ground truth", "prediction"],
                cols=2,
            )

    summary = {name: float(np.mean(values)) for name, values in metrics.items()}
    summary["score_threshold"] = threshold
    summary["avg_pred_boxes"] = float(np.mean(prediction_counts)) if prediction_counts else 0.0
    summary["avg_pred_score"] = float(np.mean(score_values)) if score_values else 0.0
    save_json(_detection_output_dir(config) / "metrics.json", summary)
    print(f"Saved detection metrics and visualizations to {_detection_output_dir(config)}")


def predict_detector(
	args,
):
    config = load_config(args.profile)
    device = resolve_device(args.device)
    model = _load_model(config, device)
    transform = build_detection_image_transform(int(config.data["detector"]["image_size"]))
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    boxes, _ = filter_predictions(output, _score_threshold(config))
    vis = draw_boxes(image_path, boxes, color="yellow")
    out_path = ensure_dir(_detection_output_dir(config) / "predictions") / f"{image_path.stem}.png"
    vis.save(out_path)
    print(f"Saved prediction visualization to {out_path}")
