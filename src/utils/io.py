import json
from pathlib import Path
from typing import Iterable


def ensure_dir(
	path: Path,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_jsonl(
	path: Path,
	rows: Iterable[dict],
):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(
	path: Path,
) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(
	path: Path,
	payload: dict,
):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
