from config import load_config
from data.transforms import build_detection_image_transform, build_retrieval_eval_transform
from detection.eval import filter_predictions
from detection.utils import _load_model as load_detector_model, _score_threshold
from retrieval.backends import build_backend
from retrieval.datasets import require_retrieval_manifests
from retrieval.utils import _encode_rows, _gather_encoded_rows, _load_encoder
from utils import cleanup_distributed, ensure_dir, save_json, setup_distributed
from utils.device import resolve_device
from utils.visualization import draw_boxes, save_image_grid
from .utils import (
    _demo_selection_key,
    _load_demo_rankings,
    _quality_from_records,
    _ranked_records,
    _resolve_demo_image_path,
    _selected_demo_items,
    _unique_display_records,
)

from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


def run_demo(args):
    config = load_config(args.profile)
    dist_state, device = setup_distributed(args.device)
    try:
        base_rows, query_rows = require_retrieval_manifests(config)
        retrieval_encoder = _load_encoder(config, device)
        retrieval_transform = build_retrieval_eval_transform(retrieval_encoder.input_size)
        batch_size = int(config.data["retrieval"]["batch_size"])
        local_base_rows = base_rows[dist_state.rank::dist_state.world_size]
        local_query_rows = query_rows[dist_state.rank::dist_state.world_size]
        base_payload = _encode_rows(retrieval_encoder, local_base_rows, retrieval_transform, batch_size, device)
        query_payload = _encode_rows(retrieval_encoder, local_query_rows, retrieval_transform, batch_size, device)
        base_result = _gather_encoded_rows(base_payload, dist_state)
        query_result = _gather_encoded_rows(query_payload, dist_state)
        if not dist_state.is_main:
            return
        base_embeddings, base_ids, _, _ = base_result
        query_embeddings, query_ids, query_labels, query_paths = query_result
        detector = load_detector_model(config, device)
        backend_name = args.index_backend or config.data["index"]["backend"]
        backend = build_backend(backend_name)
        backend.build_index(base_embeddings, base_ids)
        topk = int(args.topk or config.data["retrieval"]["topk"])
        search_topk = min(max(topk * 4, topk), len(base_ids))
        rankings = backend.search(query_embeddings, topk=search_topk)

        base_lookup = {row["id"]: row for row in base_rows}
        base_embedding_lookup = {
            sample_id: embedding for sample_id, embedding in zip(base_ids, base_embeddings, strict=True)
        }
        by_class: dict[str, list[dict]] = defaultdict(list)
        ranking_records: list[dict] = []
        for query_id, query_label, query_path, ranked_ids in zip(
            query_ids,
            query_labels,
            query_paths,
            rankings,
            strict=True,
        ):
            query_order = len(ranking_records)
            row = {
                "id": query_id,
                "label": query_label,
                "image_path": query_path,
            }
            top_records = _ranked_records(
                query_label,
                ranked_ids,
                base_lookup,
                base_embedding_lookup,
                query_embeddings[query_order],
                search_topk,
            )
            display_records = _unique_display_records(query_path, top_records, topk)
            quality = _quality_from_records(query_label, display_records, topk)
            record = {
                "query_order": query_order,
                "query": row,
                "quality": quality,
                "topk": top_records,
                "display_topk": display_records,
            }
            ranking_records.append(record)
            by_class[row["label"]].append(record)

        transform = build_detection_image_transform(int(config.data["detector"]["image_size"]))
        threshold = _score_threshold(config)
        out_dir = ensure_dir(config.paths.output_root / config.profile / "demo")
        queries_per_class = int(config.data["output"]["demo_queries_per_class"])

        demo_records: list[dict] = []

        with torch.no_grad():
            for class_name, items in sorted(by_class.items()):
                selected_items = sorted(items, key=_demo_selection_key, reverse=True)[:queries_per_class]
                for sample_idx, record in enumerate(selected_items, start=1):
                    query_row = record["query"]
                    images = [Image.open(query_row["image_path"]).convert("RGB")]
                    titles = [f"query: {class_name}"]
                    for top_record in record["display_topk"]:
                        rank = top_record["rank"]
                        image_path = Path(top_record["image_path"])
                        image = Image.open(image_path).convert("RGB")
                        tensor = transform(image).to(device)
                        output = detector([tensor])[0]
                        pred_boxes, _ = filter_predictions(output, threshold)
                        vis = draw_boxes(image_path, pred_boxes, color="yellow")
                        images.append(vis)
                        titles.append(f"top{rank}: {top_record['label']}")
                    save_image_grid(out_dir / f"{class_name}_{sample_idx:02d}.png", images, titles, cols=min(3, len(images)))
                    demo_records.append(
                        {
                            "output": f"{class_name}_{sample_idx:02d}.png",
                            "query": query_row,
                            "quality": record["quality"],
                            "topk": record["display_topk"],
                            "raw_query_order": record["query_order"],
                        }
                    )

        save_json(
            out_dir / "rankings.json",
            {
                "items": demo_records,
                "rankings": ranking_records,
                "selection": {
                    "queries_per_class": queries_per_class,
                    "display_topk": topk,
                    "search_topk": search_topk,
                    "preferred_full_match_at": topk,
                    "deduplicate_display_images": True,
                    "sort_order": [
                        "top5_all_match",
                        "top5_hits",
                        "top1_match",
                        "top5_mean_score",
                        "top1_score",
                    ],
                },
            },
        )
        print(f"Saved demo boards to {out_dir}")
    finally:
        cleanup_distributed(dist_state)


def run_test_demo(args):
    config = load_config(args.profile)
    device = resolve_device(args.device)
    detector = load_detector_model(config, device)
    transform = build_detection_image_transform(int(config.data["detector"]["image_size"]))
    threshold = _score_threshold(config)
    topk = int(args.topk or config.data["retrieval"]["topk"])
    examples = [name.strip() for name in args.examples.split(",") if name.strip()]
    if not examples:
        raise ValueError("At least one example must be provided with --examples.")

    rankings_path = config.paths.output_root / config.profile / "demo" / "rankings.json"
    rankings = _load_demo_rankings(rankings_path)
    selected_items = _selected_demo_items(rankings, examples)
    out_dir = ensure_dir(config.paths.output_root / config.profile / "test_demo")
    for stale_path in out_dir.glob("*.png"):
        stale_path.unlink()
    stale_rankings_path = out_dir / "rankings.json"
    if stale_rankings_path.exists():
        stale_rankings_path.unlink()
    project_root = Path(__file__).resolve().parents[2]
    written_items: list[dict] = []

    with torch.no_grad():
        for item in selected_items:
            query = item["query"]
            query_path = _resolve_demo_image_path(query["image_path"], project_root)
            images = [Image.open(query_path).convert("RGB")]
            titles = [f"query: {query['label']}"]
            selected_topk = item["topk"][:topk]
            written_topk: list[dict] = []

            for top_record in selected_topk:
                image_path = _resolve_demo_image_path(top_record["image_path"], project_root)
                image = Image.open(image_path).convert("RGB")
                tensor = transform(image).to(device)
                output = detector([tensor])[0]
                pred_boxes, pred_scores = filter_predictions(output, threshold)
                vis = draw_boxes(image_path, pred_boxes, color="yellow")
                images.append(vis)
                titles.append(f"top{top_record['rank']}: {top_record['label']}")
                written_record = dict(top_record)
                written_record["image_path"] = str(image_path)
                written_record["detected_boxes"] = pred_boxes
                written_record["detected_scores"] = pred_scores
                written_topk.append(written_record)

            output_name = item["output"]
            save_image_grid(out_dir / output_name, images, titles, cols=min(3, len(images)))
            written_query = dict(query)
            written_query["image_path"] = str(query_path)
            written_items.append(
                {
                    "output": output_name,
                    "query": written_query,
                    "quality": item.get("quality", {}),
                    "topk": written_topk,
                    "source_rankings": str(rankings_path),
                }
            )

    save_json(
        out_dir / "rankings.json",
        {
            "items": written_items,
            "selection": {
                "source": str(rankings_path),
                "examples": examples,
                "display_topk": topk,
                "score_threshold": threshold,
                "purpose": "lightweight recording demo using saved retrieval rankings and live text detection",
            },
        },
    )
    print(f"Saved lightweight test demo boards to {out_dir}")
