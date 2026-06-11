import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))

from data.prepare import prepare_data
from detection.eval import evaluate_detector, predict_detector
from detection.train import train_detector
from demo.run import run_demo, run_test_demo
from retrieval.eval import evaluate_retrieval
from retrieval.train import train_retrieval


def build_parser(
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BJTU2026 campus vision retrieval and text detection.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Prepare BJTU2026 retrieval and detection manifests.")
    prepare.add_argument("--profile", choices=["mini", "full"], default="mini")
    prepare.add_argument(
        "--archive",
        default=None,
        help="Path to BJTU2026dataset.zip. Defaults to the path in config.",
    )
    prepare.add_argument("--no-download", action="store_true", help="Do not extract the BJTU2026 archive automatically.")
    prepare.set_defaults(func=prepare_data)

    for name, func in [
        ("train-retrieval", train_retrieval),
        ("eval-retrieval", evaluate_retrieval),
        ("train-detector", train_detector),
        ("eval-detector", evaluate_detector),
        ("demo", run_demo),
        ("test-demo", run_test_demo),
    ]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--profile", choices=["mini", "full"], default="mini")
        sub.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
        if name in {"train-retrieval", "train-detector"}:
            sub.add_argument("--epochs", type=int, default=None)
            sub.add_argument("--max-batches", type=int, default=None)
        if name in {"eval-retrieval", "demo", "test-demo"}:
            sub.add_argument("--topk", type=int, default=None)
        if name == "train-retrieval":
            sub.add_argument(
                "--no-pretrained",
                action="store_true",
                help="Initialize the retrieval backbone without downloading pretrained DINOv2 weights.",
            )
        if name == "train-detector":
            sub.add_argument(
                "--no-pretrained",
                action="store_true",
                help="Initialize the detector without downloading torchvision COCO weights.",
            )
        if name in {"eval-retrieval", "demo"}:
            sub.add_argument("--index-backend", choices=["sklearn", "faiss"], default=None)
        if name == "test-demo":
            sub.add_argument(
                "--examples",
                default="mh_02.png,nm_01.png,sjz_01.png",
                help="Comma-separated demo board names selected from outputs/<profile>/demo/rankings.json.",
            )
        sub.set_defaults(func=func)

    predict = subparsers.add_parser("predict-detector")
    predict.add_argument("--profile", choices=["mini", "full"], default="mini")
    predict.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    predict.add_argument("--image", required=True)
    predict.set_defaults(func=predict_detector)
    return parser


def main(
):
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
