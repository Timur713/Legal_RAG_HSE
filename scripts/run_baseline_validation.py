from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

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
    parser = argparse.ArgumentParser(description="Run the TF-IDF baseline with local strict CV and holdout evaluation.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--experiment-name",
        default="baseline_validation",
        help="Prefix for saved metrics and prediction artifacts.",
    )
    parser.add_argument(
        "--validation-splits",
        default="data/processed/validation/validation_splits.csv",
        help="Path to validation_splits.csv created by scripts/make_validation_splits.py.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many documents to retrieve per question.",
    )
    parser.add_argument(
        "--save-test-predictions",
        action="store_true",
        help="Also fit on the full train corpus and save test predictions for submission generation.",
    )
    return parser.parse_args()


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_validation_splits(path_value: str) -> pd.DataFrame:
    path = resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(
            f"Validation splits file does not exist: {path}. Run scripts/make_validation_splits.py first."
        )
    splits = pd.read_csv(path)
    required_columns = {"qid", "strict_holdout_role", "strict_cv_fold"}
    missing = required_columns - set(splits.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Validation splits are missing columns: {missing_columns}")
    splits["qid"] = splits["qid"].astype(str)
    return splits


def evaluate_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[float, pd.DataFrame]:
    retriever = TfidfRetriever(top_k=top_k).fit(documents)
    ranked_doc_ids, ranked_scores = retriever.retrieve(eval_df["question"].tolist(), top_k=top_k)
    metric_value = recall_at_k(eval_df["gold_doc_id"].tolist(), ranked_doc_ids, k=top_k)
    predictions = build_ranked_predictions(eval_df["qid"].tolist(), ranked_doc_ids, ranked_scores)
    return metric_value, predictions


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)
    ensure_output_dirs(paths)

    documents = load_documents(paths)
    train = load_train(paths)
    test = load_test(paths)
    splits = load_validation_splits(args.validation_splits)

    merged = train.merge(
        splits[["qid", "strict_holdout_role", "strict_cv_fold"]],
        on="qid",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(train):
        raise ValueError("Validation splits do not align with train.csv by qid.")

    dev = merged[merged["strict_holdout_role"] == "dev"].copy()
    holdout = merged[merged["strict_holdout_role"] == "holdout"].copy()

    fold_metrics: list[dict[str, object]] = []
    fold_predictions: list[pd.DataFrame] = []
    for fold in sorted(dev["strict_cv_fold"].dropna().astype(int).unique()):
        fold_train = dev[dev["strict_cv_fold"] != fold].copy()
        fold_val = dev[dev["strict_cv_fold"] == fold].copy()
        metric_value, predictions = evaluate_split(
            fold_train,
            fold_val,
            documents,
            top_k=args.top_k,
        )
        fold_metrics.append(
            {
                "split": "strict_cv",
                "fold": int(fold),
                "num_train_queries": int(len(fold_train)),
                "num_val_queries": int(len(fold_val)),
                "num_val_gold_docs": int(fold_val["gold_doc_id"].nunique()),
                f"recall@{args.top_k}": float(metric_value),
            }
        )
        predictions["split"] = "strict_cv"
        predictions["fold"] = int(fold)
        fold_predictions.append(predictions)

    holdout_metric, holdout_predictions = evaluate_split(
        dev,
        holdout,
        documents,
        top_k=args.top_k,
    )
    holdout_predictions["split"] = "strict_holdout"
    holdout_predictions["fold"] = pd.NA

    metric_name = f"recall@{args.top_k}"
    cv_scores = [row[metric_name] for row in fold_metrics]
    summary_metrics = {
        "experiment_name": args.experiment_name,
        "metric_name": metric_name,
        "strict_cv_mean": float(sum(cv_scores) / len(cv_scores)),
        "strict_cv_std": float(pd.Series(cv_scores).std(ddof=0)),
        "strict_holdout": float(holdout_metric),
        "num_documents": int(len(documents)),
        "num_train_queries": int(len(train)),
        "num_dev_queries": int(len(dev)),
        "num_holdout_queries": int(len(holdout)),
        "top_k": int(args.top_k),
        "validation_splits": str(resolve_path(args.validation_splits)),
        "fold_metrics": fold_metrics,
    }

    predictions_dir = paths.outputs_dir / "predictions"
    metrics_dir = paths.outputs_dir / "metrics"

    cv_predictions_path = predictions_dir / f"{args.experiment_name}_strict_cv_predictions.csv"
    holdout_predictions_path = predictions_dir / f"{args.experiment_name}_strict_holdout_predictions.csv"
    metrics_path = metrics_dir / f"{args.experiment_name}_metrics.json"

    pd.concat(fold_predictions, ignore_index=True).to_csv(cv_predictions_path, index=False)
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    metrics_path.write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved strict CV predictions to {cv_predictions_path}")
    print(f"Saved strict holdout predictions to {holdout_predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"{metric_name} strict_cv_mean={summary_metrics['strict_cv_mean']:.4f}")
    print(f"{metric_name} strict_cv_std={summary_metrics['strict_cv_std']:.4f}")
    print(f"{metric_name} strict_holdout={summary_metrics['strict_holdout']:.4f}")

    if args.save_test_predictions:
        retriever = TfidfRetriever(top_k=args.top_k).fit(documents)
        test_doc_ids, test_scores = retriever.retrieve(test["question"].tolist(), top_k=args.top_k)
        test_predictions = build_ranked_predictions(test["qid"].tolist(), test_doc_ids, test_scores)
        test_predictions_path = predictions_dir / f"{args.experiment_name}_test_predictions.csv"
        test_predictions.to_csv(test_predictions_path, index=False)
        print(f"Saved test predictions to {test_predictions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
