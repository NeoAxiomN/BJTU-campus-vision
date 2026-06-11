from utils import ensure_dir
from utils.distributed import DistributedState
from .datasets import RetrievalEncodeDataset
from .model import DINOv2GeMEncoder

from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def _retrieval_output_dir(
	config,
) -> Path:
    return ensure_dir(config.paths.output_root / config.profile / "retrieval")


def _load_encoder(
	config,
	device: torch.device,
) -> DINOv2GeMEncoder:
    checkpoint_path = _retrieval_output_dir(config) / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Retrieval checkpoint is missing. Run `train-retrieval --profile "
            f"{config.profile}` first."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder = DINOv2GeMEncoder(checkpoint["model_name"], pretrained=False)
    encoder.load_state_dict(checkpoint["encoder"])
    return encoder.to(device).eval()


def _encode_rows(
	encoder,
	rows: list[dict],
	transform,
	batch_size: int,
	device: torch.device,
):
    dataset = RetrievalEncodeDataset(rows, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, batch_ids, batch_labels, batch_paths in loader:
            features = encoder(images.to(device)).cpu().numpy()
            embeddings.append(features)
            ids.extend(batch_ids)
            labels.extend(batch_labels)
            paths.extend(batch_paths)
    return np.concatenate(embeddings, axis=0), ids, labels, paths


def _gather_encoded_rows(
	payload: tuple[np.ndarray, list[str], list[str], list[str]],
	state: DistributedState,
) -> tuple[np.ndarray, list[str], list[str], list[str]] | None:
    if not state.enabled:
        return payload
    gathered = [None for _ in range(state.world_size)] if state.is_main else None
    dist.gather_object(payload, gathered, dst=0)
    if not state.is_main:
        return None
    embeddings: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[str] = []
    paths: list[str] = []
    for item in gathered:
        batch_embeddings, batch_ids, batch_labels, batch_paths = item
        embeddings.append(batch_embeddings)
        ids.extend(batch_ids)
        labels.extend(batch_labels)
        paths.extend(batch_paths)
    return np.concatenate(embeddings, axis=0), ids, labels, paths


def _precision_curve(
	query_label: str,
	ranked_ids: list[str],
	id_to_label: dict[str, str],
	topk: int,
) -> list[float]:
    values: list[float] = []
    hits = 0
    for idx, sample_id in enumerate(ranked_ids[:topk], start=1):
        if id_to_label[sample_id] == query_label:
            hits += 1
        values.append(hits / idx)
    return values
