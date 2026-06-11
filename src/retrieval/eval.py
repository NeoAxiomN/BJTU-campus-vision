from config import load_config
from data.transforms import build_retrieval_eval_transform
from utils import cleanup_distributed, ensure_dir, save_json, setup_distributed
from utils.visualization import save_image_grid, save_precision_plot
from .backends import build_backend
from .datasets import require_retrieval_manifests
from .utils import _encode_rows, _gather_encoded_rows, _load_encoder, _precision_curve, _retrieval_output_dir

from collections import defaultdict

import numpy as np
from PIL import Image


def evaluate_retrieval(
	args,
):
    config = load_config(args.profile)
    dist_state, device = setup_distributed(args.device)
    try:
        base_rows, query_rows = require_retrieval_manifests(config)
        encoder = _load_encoder(config, device)
        transform = build_retrieval_eval_transform(encoder.input_size)
        batch_size = int(config.data["retrieval"]["batch_size"])
        local_base_rows = base_rows[dist_state.rank::dist_state.world_size]
        local_query_rows = query_rows[dist_state.rank::dist_state.world_size]
        base_payload = _encode_rows(encoder, local_base_rows, transform, batch_size, device)
        query_payload = _encode_rows(encoder, local_query_rows, transform, batch_size, device)
        base_result = _gather_encoded_rows(base_payload, dist_state)
        query_result = _gather_encoded_rows(query_payload, dist_state)
        if not dist_state.is_main:
            return
        base_embeddings, base_ids, base_labels, base_paths = base_result
        query_embeddings, query_ids, query_labels, query_paths = query_result
        backend_name = args.index_backend or config.data["index"]["backend"]
        backend = build_backend(backend_name)
        topk = int(args.topk or config.data["retrieval"]["topk"])
        backend.build_index(base_embeddings, base_ids)
        rankings = backend.search(query_embeddings, topk=topk)

        id_to_label = {sample_id: label for sample_id, label in zip(base_ids, base_labels, strict=True)}
        id_to_path = {sample_id: path for sample_id, path in zip(base_ids, base_paths, strict=True)}
        ranking_records = []
        per_class: dict[str, list[list[float]]] = defaultdict(list)
        for query_id, query_label, query_path, ranked_ids in zip(
            query_ids,
            query_labels,
            query_paths,
            rankings,
            strict=True,
        ):
            per_class[query_label].append(_precision_curve(query_label, ranked_ids, id_to_label, topk))
            ranking_records.append(
                {
                    "query": {
                        "id": query_id,
                        "label": query_label,
                        "image_path": query_path,
                    },
                    "topk": [
                        {
                            "rank": rank,
                            "id": sample_id,
                            "label": id_to_label[sample_id],
                            "image_path": id_to_path[sample_id],
                        }
                        for rank, sample_id in enumerate(ranked_ids[:topk], start=1)
                    ],
                }
            )

        out_dir = _retrieval_output_dir(config)
        plot_dir = ensure_dir(out_dir / "plots")
        ks = list(range(1, topk + 1))
        metrics: dict[str, dict[str, float]] = {}
        for class_name, curves in sorted(per_class.items()):
            mean_curve = np.mean(np.asarray(curves), axis=0)
            save_precision_plot(plot_dir / f"{class_name}.png", class_name, ks, mean_curve.tolist())
            metrics[class_name] = {f"p@{k}": float(v) for k, v in zip(ks, mean_curve, strict=True)}

        example_dir = ensure_dir(out_dir / "examples")
        seen_classes: set[str] = set()
        for query_id, query_label, query_path, ranked_ids in zip(query_ids, query_labels, query_paths, rankings, strict=True):
            if query_label in seen_classes:
                continue
            seen_classes.add(query_label)
            images = [Image.open(query_path).convert("RGB")]
            titles = [f"query: {query_label}"]
            for rank, sample_id in enumerate(ranked_ids[: min(5, topk)], start=1):
                images.append(Image.open(id_to_path[sample_id]).convert("RGB"))
                titles.append(f"top{rank}: {id_to_label[sample_id]}")
            save_image_grid(example_dir / f"{query_label}.png", images, titles, cols=3)

        macro = {}
        for k in ks:
            macro[f"p@{k}"] = float(np.mean([values[f"p@{k}"] for values in metrics.values()]))
        save_json(
            out_dir / "metrics.json",
            {
                "per_class": metrics,
                "macro": macro,
                "backend": backend_name,
                "distributed": dist_state.enabled,
                "world_size": dist_state.world_size,
            },
        )
        save_json(out_dir / "rankings.json", {"items": ranking_records})
        print(f"Saved retrieval metrics and plots to {out_dir}")
    finally:
        cleanup_distributed(dist_state)
