from config import load_config
from utils import ensure_dir, save_jsonl, set_seed
from .manifests import (
    base_manifest_path,
    det_train_manifest_path,
    det_val_manifest_path,
    manifest_dir,
    query_manifest_path,
)
from .utils import (
    NEGATIVE_LABEL,
    _archive_path,
    _build_detection_rows,
    _build_retrieval_rows,
    _dataset_root,
    _ensure_dataset,
)

from pathlib import Path


def prepare_data(args):
    config = load_config(args.profile)
    seed = int(config.data["runtime"]["seed"])
    set_seed(seed)

    dataset_root = _ensure_dataset(
        _dataset_root(config.paths.raw_root, config.data),
        Path(args.archive).expanduser() if getattr(args, "archive", None) else _archive_path(config.data),
        allow_extract=not args.no_download,
    )
    ensure_dir(manifest_dir(config))
    base_rows, query_rows = _build_retrieval_rows(dataset_root, config.data, args.profile, seed)
    det_train_rows, det_val_rows = _build_detection_rows(dataset_root, config.data, args.profile, seed)

    save_jsonl(base_manifest_path(config), base_rows)
    save_jsonl(query_manifest_path(config), query_rows)
    save_jsonl(det_train_manifest_path(config), det_train_rows)
    save_jsonl(det_val_manifest_path(config), det_val_rows)

    bjtubase_count = sum(1 for row in base_rows if row["label"] != NEGATIVE_LABEL)
    negative_count = len(base_rows) - bjtubase_count
    print(f"Prepared BJTU2026 profile={args.profile}")
    print(f"Dataset root: {dataset_root}")
    print(f"Retrieval base BJTU/util: {bjtubase_count}/{negative_count}")
    print(f"Retrieval query: {len(query_rows)}")
    print(f"Detection train/val: {len(det_train_rows)}/{len(det_val_rows)}")
