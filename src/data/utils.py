from utils import ensure_dir

import json
import random
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
NEGATIVE_LABEL = "__negative__"


def _dataset_root(
	raw_root: Path,
	config: dict,
) -> Path:
    return raw_root / str(config["dataset"].get("raw_dir", "BJTU2026"))


def _archive_path(
	config: dict,
) -> Path:
    return Path(str(config["dataset"].get("archive_path", "/home/junyi/BJTU2026dataset.zip"))).expanduser()


def _ensure_dataset(
	root: Path,
	archive_path: Path,
	allow_extract: bool,
) -> Path:
    if (root / "image_retrieval").exists() and (root / "object_detection").exists():
        return root
    if not allow_extract:
        raise FileNotFoundError(
            "BJTU2026 dataset is missing. Place it under "
            f"{root} or rerun without --no-download to extract {archive_path}."
        )
    if not archive_path.exists():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")
    ensure_dir(root)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    if not (root / "image_retrieval").exists() or not (root / "object_detection").exists():
        raise RuntimeError(f"Extracted archive but BJTU2026 folders were not found under {root}")
    return root


def _image_files(
	root: Path,
) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _is_readable_image(
	path: Path,
) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def _landmark_from_stem(
	stem: str,
) -> str:
    if "-" not in stem:
        return NEGATIVE_LABEL
    return stem.split("-", 1)[0].lower()


def _limit_by_class(
	rows: list[dict],
	limit: int | None,
	seed: int,
) -> list[dict]:
    if limit is None:
        return rows
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    selected: list[dict] = []
    for label, items in sorted(groups.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        selected.extend(sorted(shuffled[:limit], key=lambda item: item["id"]))
    return selected


def _limit_rows(
	rows: list[dict],
	limit: int | None,
	seed: int,
) -> list[dict]:
    if limit is None or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda item: item["id"])


def _build_retrieval_rows(
	dataset_root: Path,
	config: dict,
	profile: str,
	seed: int,
) -> tuple[list[dict], list[dict]]:
    retrieval_root = dataset_root / "image_retrieval"
    profile_config = dict(config["dataset"].get(profile, {}))
    base_per_class = profile_config.get("base_per_class")
    query_per_class = profile_config.get("query_per_class")
    negative_count = profile_config.get("negative_count")
    if profile == "full":
        base_per_class = None
        query_per_class = None
        negative_count = None

    base_rows: list[dict] = []
    skipped_bad_images = 0
    bjtubase_dir = retrieval_root / "base" / "BJTU"
    for image_path in _image_files(bjtubase_dir):
        if not _is_readable_image(image_path):
            skipped_bad_images += 1
            continue
        label = _landmark_from_stem(image_path.stem)
        base_rows.append(
            {
                "id": f"bjtubase_{image_path.stem}",
                "label": label,
                "image_path": str(image_path.resolve()),
                "source_split": "base",
                "source_group": "BJTU",
            }
        )
    base_rows = _limit_by_class(base_rows, None if base_per_class is None else int(base_per_class), seed)

    negative_rows: list[dict] = []
    util_dir = retrieval_root / "base" / "util_pic"
    for image_path in _image_files(util_dir):
        if not _is_readable_image(image_path):
            skipped_bad_images += 1
            continue
        negative_rows.append(
            {
                "id": f"util_{image_path.stem}",
                "label": NEGATIVE_LABEL,
                "image_path": str(image_path.resolve()),
                "source_split": "base",
                "source_group": "util_pic",
            }
        )
    negative_rows = _limit_rows(negative_rows, None if negative_count is None else int(negative_count), seed + 1)
    base_rows = sorted(base_rows + negative_rows, key=lambda item: item["id"])

    query_rows: list[dict] = []
    query_dir = retrieval_root / "query"
    for image_path in _image_files(query_dir):
        if not _is_readable_image(image_path):
            skipped_bad_images += 1
            continue
        label = _landmark_from_stem(image_path.stem)
        query_rows.append(
            {
                "id": f"query_{image_path.stem}",
                "label": label,
                "image_path": str(image_path.resolve()),
                "source_split": "query",
            }
        )
    query_rows = _limit_by_class(query_rows, None if query_per_class is None else int(query_per_class), seed + 2)
    query_rows = sorted(query_rows, key=lambda item: item["id"])
    if skipped_bad_images:
        print(f"Skipped unreadable retrieval images: {skipped_bad_images}")
    return base_rows, query_rows


def _resolve_labelme_image(
	data_dir: Path,
	payload: dict,
	json_path: Path,
) -> Path | None:
    image_path = payload.get("imagePath")
    candidates: list[Path] = []
    if image_path:
        candidates.append(data_dir / str(image_path))
    for suffix in IMAGE_SUFFIXES:
        candidates.append(json_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _rectangle_to_box(
	points: list[list[float]],
	width: int,
	height: int,
) -> list[int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x1 = max(0, min(width, int(round(min(xs)))))
    y1 = max(0, min(height, int(round(min(ys)))))
    x2 = max(0, min(width, int(round(max(xs)))))
    y2 = max(0, min(height, int(round(max(ys)))))
    return [x1, y1, x2, y2]


def _build_detection_rows(
	dataset_root: Path,
	config: dict,
	profile: str,
	seed: int,
) -> tuple[list[dict], list[dict]]:
    data_dir = dataset_root / "object_detection" / "data"
    rows: list[dict] = []
    for json_path in sorted(data_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        image_path = _resolve_labelme_image(data_dir, payload, json_path)
        if image_path is None:
            continue
        width = int(payload.get("imageWidth") or 0)
        height = int(payload.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            with Image.open(image_path) as image:
                width, height = image.size
        boxes: list[list[int]] = []
        texts: list[str] = []
        for shape in payload.get("shapes", []):
            points = shape.get("points") or []
            if len(points) < 2:
                continue
            box = _rectangle_to_box(points, width, height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            texts.append(str(shape.get("label", "")))
        if not boxes:
            continue
        rows.append(
            {
                "id": json_path.stem,
                "label": _landmark_from_stem(json_path.stem),
                "image_path": str(image_path.resolve()),
                "boxes": boxes,
                "texts": texts,
                "width": width,
                "height": height,
            }
        )

    profile_config = dict(config["dataset"].get(profile, {}))
    detection_per_class = profile_config.get("detection_per_class")
    if profile != "full" and detection_per_class is not None:
        rows = _limit_by_class(rows, int(detection_per_class), seed + 3)
    rng = random.Random(seed + 4)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["label"], []).append(row)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for label, items in sorted(groups.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        split_at = max(1, int(0.8 * len(shuffled)))
        if len(shuffled) > 1:
            split_at = min(split_at, len(shuffled) - 1)
        train_rows.extend(shuffled[:split_at])
        val_rows.extend(shuffled[split_at:])
    return sorted(train_rows, key=lambda item: item["id"]), sorted(val_rows, key=lambda item: item["id"])
