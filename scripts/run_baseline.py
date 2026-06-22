from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.baseline import TfidfRetriever, build_ranked_predictions  # noqa: E402
from legal_rag.data import (  # noqa: E402
    ensure_output_dirs,
    load_documents,
    load_paths_config,
    load_test,
    load_train,
)
from legal_rag.evaluation import recall_at_k  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TF-IDF baseline and save outputs.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--experiment-name",
        default="baseline",
        help="Prefix for saved metrics, predictions, and submission artifacts.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many documents to retrieve per question.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)
    ensure_output_dirs(paths)

    documents = load_documents(paths)
    train = load_train(paths)
    test = load_test(paths)

    retriever = TfidfRetriever(top_k=args.top_k).fit(documents)

    train_doc_ids, train_scores = retriever.retrieve(train["question"].tolist(), top_k=args.top_k)
    test_doc_ids, test_scores = retriever.retrieve(test["question"].tolist(), top_k=args.top_k)

    train_predictions = build_ranked_predictions(train["qid"].tolist(), train_doc_ids, train_scores)
    test_predictions = build_ranked_predictions(test["qid"].tolist(), test_doc_ids, test_scores)

    metric_name = f"recall@{args.top_k}"
    metrics = {
        "experiment_name": args.experiment_name,
        "metric_name": metric_name,
        "metric_value": recall_at_k(train["gold_doc_id"].tolist(), train_doc_ids, k=args.top_k),
        "num_documents": int(len(documents)),
        "num_train_queries": int(len(train)),
        "num_test_queries": int(len(test)),
        "top_k": int(args.top_k),
    }

    predictions_dir = paths.outputs_dir / "predictions"
    metrics_dir = paths.outputs_dir / "metrics"

    train_predictions_path = predictions_dir / f"{args.experiment_name}_train_predictions.csv"
    test_predictions_path = predictions_dir / f"{args.experiment_name}_test_predictions.csv"
    metrics_path = metrics_dir / f"{args.experiment_name}_metrics.json"

    train_predictions.to_csv(train_predictions_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved train predictions to {train_predictions_path}")
    print(f"Saved test predictions to {test_predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"{metric_name}={metrics['metric_value']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
