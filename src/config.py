from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ProjectPaths:
    project_root: Path
    raw_root: Path
    processed_root: Path
    output_root: Path


@dataclass(slots=True)
class LoadedConfig:
    profile: str
    project_root: Path
    data: dict[str, Any]
    paths: ProjectPaths


def load_config(profile: str) -> LoadedConfig:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / f"{profile}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config profile not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    paths = ProjectPaths(
        project_root=project_root,
        raw_root=project_root / data["paths"]["raw_root"],
        processed_root=project_root / data["paths"]["processed_root"],
        output_root=project_root / data["paths"]["output_root"],
    )
    return LoadedConfig(profile=profile, project_root=project_root, data=data, paths=paths)
