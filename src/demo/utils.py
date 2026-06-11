import hashlib
import json
from pathlib import Path

from PIL import Image


def _load_demo_rankings(
	path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing demo rankings: {path}. Run `bjtu-campus-vision demo --profile full` first, "
            "or use the generated rankings committed under outputs/<profile>/demo/."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_demo_image_path(
	image_path: str,
	project_root: Path,
) -> Path:
    path = Path(image_path)
    if path.exists():
        return path

    parts = path.parts
    if "data" in parts:
        data_index = parts.index("data")
        candidate = project_root.joinpath(*parts[data_index:])
        if candidate.exists():
            return candidate
    return path


def _selected_demo_items(
	rankings: dict,
	examples: list[str],
) -> list[dict]:
    item_lookup = {item["output"]: item for item in rankings.get("items", [])}
    missing = [example for example in examples if example not in item_lookup]
    if missing:
        available = ", ".join(sorted(item_lookup))
        raise ValueError(f"Missing examples in rankings.json: {missing}. Available examples: {available}")
    return [item_lookup[example] for example in examples]


def _image_fingerprint(
	image_path: str,
) -> str:
    image = Image.open(image_path).convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    return hashlib.sha256(image.tobytes()).hexdigest()


def _unique_display_records(
	query_path: str,
	top_records: list[dict],
	display_topk: int,
) -> list[dict]:
    seen = {_image_fingerprint(query_path)}
    display_records: list[dict] = []
    for record in top_records:
        fingerprint = _image_fingerprint(record["image_path"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        display_records.append(record)
        if len(display_records) >= display_topk:
            break
    return display_records


def _quality_from_records(
	query_label: str,
	records: list[dict],
	display_topk: int,
) -> dict:
    selected = records[:display_topk]
    top_scores = [float(record["score"]) for record in selected]
    top_hits = sum(1 for record in selected if record["label"] == query_label)
    top1 = selected[0] if selected else {}
    return {
        "top1_match": bool(top1.get("match", False)),
        "top5_all_match": len(selected) == display_topk and top_hits == display_topk,
        "top5_hits": top_hits,
        "topk_hits": sum(1 for record in records if record["label"] == query_label),
        "top1_score": float(top1.get("score", 0.0)),
        "top5_mean_score": float(sum(top_scores) / len(top_scores)) if top_scores else 0.0,
        "unique_results": len(selected),
    }


def _ranked_records(
	query_label: str,
	ranked_ids: list[str],
	base_lookup: dict[str, dict],
	base_embedding_lookup: dict[str, object],
	query_embedding,
	topk: int,
) -> list[dict]:
    top_records: list[dict] = []
    for rank, sample_id in enumerate(ranked_ids[:topk], start=1):
        base_row = base_lookup[sample_id]
        score = float(query_embedding @ base_embedding_lookup[sample_id])
        is_match = base_row["label"] == query_label
        top_records.append(
            {
                "rank": rank,
                "id": sample_id,
                "label": base_row["label"],
                "image_path": base_row["image_path"],
                "score": score,
                "match": is_match,
            }
        )
    return top_records


def _demo_selection_key(
	record: dict,
) -> tuple:
    quality = record["quality"]
    return (
        quality["top5_all_match"],
        quality["top5_hits"],
        quality["top1_match"],
        quality["top5_mean_score"],
        quality["top1_score"],
        -record["query_order"],
    )
