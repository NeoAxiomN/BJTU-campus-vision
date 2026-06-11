from config import LoadedConfig

from pathlib import Path


def profile_root(config: LoadedConfig) -> Path:
    return config.paths.processed_root / config.profile


def manifest_dir(config: LoadedConfig) -> Path:
    return profile_root(config) / "manifests"


def base_manifest_path(config: LoadedConfig) -> Path:
    return manifest_dir(config) / "base_manifest.jsonl"


def query_manifest_path(config: LoadedConfig) -> Path:
    return manifest_dir(config) / "query_manifest.jsonl"


def det_train_manifest_path(config: LoadedConfig) -> Path:
    return manifest_dir(config) / "det_train.jsonl"


def det_val_manifest_path(config: LoadedConfig) -> Path:
    return manifest_dir(config) / "det_val.jsonl"
