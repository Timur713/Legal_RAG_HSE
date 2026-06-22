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

from legal_rag.data import (  # noqa: E402
    ensure_output_dirs,
    load_documents,
    load_paths_config,
    load_test,
    load_train,
)
from legal_rag.evaluation import recall_at_k  # noqa: E402
from legal_rag.retrieval import (  # noqa: E402
    AVAILABLE_RETRIEVERS,
    build_ranked_predictions,
    create_retriever,
    get_retriever_params,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval validation experiments with pluggable retrievers."
    )
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
        "--retriever",
        default="tfidf",
        choices=AVAILABLE_RETRIEVERS,
        help="Retriever to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Primary Recall@k metric to optimize and report at the top level.",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=None,
        help="How many documents to retrieve and save per query. Defaults to max(top-k, extra-metric-k).",
    )
    parser.add_argument(
        "--extra-metric-k",
        type=int,
        nargs="*",
        default=(),
        help="Optional extra Recall@k values to compute, for example --extra-metric-k 20.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1600,
        help="Chunk size in characters for chunked retrievers.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=None,
        help="Chunk stride in characters for chunked retrievers. Defaults to chunk-size // 2.",
    )
    parser.add_argument(
        "--use-lemmas",
        action="store_true",
        help="Apply Russian lemmatization with pymorphy3 during tokenization.",
    )
    parser.add_argument(
        "--hybrid-retrievers",
        nargs="*",
        default=(),
        help="Component retrievers for hybrid_rrf, for example: --hybrid-retrievers bm25 chunked_bm25.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant for hybrid_rrf.",
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


def get_metric_ks(args: argparse.Namespace) -> list[int]:
    metric_ks = [int(args.top_k)]
    for value in args.extra_metric_k:
        metric_k = int(value)
        if metric_k not in metric_ks:
            metric_ks.append(metric_k)

    if any(metric_k <= 0 for metric_k in metric_ks):
        raise ValueError("All metric k values must be positive.")

    return metric_ks


def get_retrieve_k(args: argparse.Namespace, metric_ks: list[int]) -> int:
    retrieve_k = int(args.retrieve_k) if args.retrieve_k is not None else max(metric_ks)
    if retrieve_k <= 0:
        raise ValueError("retrieve_k must be positive.")

    max_metric_k = max(metric_ks)
    if retrieve_k < max_metric_k:
        raise ValueError(
            f"retrieve_k={retrieve_k} is smaller than the largest requested metric k={max_metric_k}."
        )

    return retrieve_k


def validate_hybrid_args(args: argparse.Namespace) -> None:
    if args.retriever != "hybrid_rrf":
        return

    if len(args.hybrid_retrievers) < 2:
        raise ValueError("hybrid_rrf requires at least two component retrievers in --hybrid-retrievers.")
    if any(component == "hybrid_rrf" for component in args.hybrid_retrievers):
        raise ValueError("hybrid_rrf cannot include hybrid_rrf as a component retriever.")
    if args.rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")


def compute_recall_metrics(
    gold_doc_ids: list[str],
    ranked_doc_ids: list[list[str]],
    metric_ks: list[int],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_k in metric_ks:
        metrics[f"recall@{metric_k}"] = float(recall_at_k(gold_doc_ids, ranked_doc_ids, k=metric_k))
    return metrics


def build_topic_metrics(
    eval_df: pd.DataFrame,
    ranked_doc_ids: list[list[str]],
    metric_ks: list[int],
    *,
    split: str,
    fold: int | None,
) -> pd.DataFrame:
    eval_frame = eval_df.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for topic, group in eval_frame.groupby("topic", dropna=False):
        positions = group.index.tolist()
        predictions = [ranked_doc_ids[position] for position in positions]
        row: dict[str, object] = {
            "split": split,
            "fold": fold,
            "topic": str(topic),
            "num_queries": int(len(group)),
            "num_gold_docs": int(group["gold_doc_id"].nunique()),
        }
        row.update(compute_recall_metrics(group["gold_doc_id"].tolist(), predictions, metric_ks))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "fold", "topic"], kind="stable").reset_index(drop=True)


def evaluate_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    documents: pd.DataFrame,
    *,
    retriever_name: str,
    retriever_params: dict[str, object],
    retrieve_k: int,
    metric_ks: list[int],
    split: str,
    fold: int | None,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    retriever = create_retriever(retriever_name, top_k=retrieve_k, **retriever_params)
    retriever.fit(documents, train_queries=train_df)

    ranked_doc_ids, ranked_scores = retriever.retrieve(eval_df["question"].tolist(), top_k=retrieve_k)
    metrics = compute_recall_metrics(eval_df["gold_doc_id"].tolist(), ranked_doc_ids, metric_ks)
    predictions = build_ranked_predictions(eval_df["qid"].tolist(), ranked_doc_ids, ranked_scores)
    predictions["split"] = split
    predictions["fold"] = fold

    topic_metrics = build_topic_metrics(
        eval_df,
        ranked_doc_ids,
        metric_ks,
        split=split,
        fold=fold,
    )
    return metrics, predictions, topic_metrics


def aggregate_metric_summary(
    fold_metrics: list[dict[str, object]],
    holdout_metrics: dict[str, float],
    metric_ks: list[int],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric_k in metric_ks:
        metric_name = f"recall@{metric_k}"
        cv_scores = [float(row[metric_name]) for row in fold_metrics]
        summary[metric_name] = {
            "strict_cv_mean": float(sum(cv_scores) / len(cv_scores)),
            "strict_cv_std": float(pd.Series(cv_scores).std(ddof=0)),
            "strict_holdout": float(holdout_metrics[metric_name]),
        }
    return summary


def main() -> int:
    args = parse_args()
    metric_ks = get_metric_ks(args)
    retrieve_k = get_retrieve_k(args, metric_ks)
    validate_hybrid_args(args)

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

    retriever_params = get_retriever_params(
        args.retriever,
        chunk_size=args.chunk_size,
        chunk_stride=args.chunk_stride,
        use_lemmas=args.use_lemmas,
        hybrid_retrievers=args.hybrid_retrievers,
        rrf_k=args.rrf_k,
    )
    retriever_runtime_params = {
        key: value
        for key, value in retriever_params.items()
        if key != "name"
    }

    fold_metrics: list[dict[str, object]] = []
    fold_predictions: list[pd.DataFrame] = []
    topic_metrics_frames: list[pd.DataFrame] = []

    for fold in sorted(dev["strict_cv_fold"].dropna().astype(int).unique()):
        fold_train = dev[dev["strict_cv_fold"] != fold].copy()
        fold_val = dev[dev["strict_cv_fold"] == fold].copy()
        metrics, predictions, topic_metrics = evaluate_split(
            fold_train,
            fold_val,
            documents,
            retriever_name=args.retriever,
            retriever_params=retriever_runtime_params,
            retrieve_k=retrieve_k,
            metric_ks=metric_ks,
            split="strict_cv",
            fold=int(fold),
        )

        fold_row: dict[str, object] = {
            "split": "strict_cv",
            "fold": int(fold),
            "num_train_queries": int(len(fold_train)),
            "num_val_queries": int(len(fold_val)),
            "num_val_gold_docs": int(fold_val["gold_doc_id"].nunique()),
        }
        fold_row.update(metrics)
        fold_metrics.append(fold_row)
        fold_predictions.append(predictions)
        topic_metrics_frames.append(topic_metrics)

    holdout_metrics, holdout_predictions, holdout_topic_metrics = evaluate_split(
        dev,
        holdout,
        documents,
        retriever_name=args.retriever,
        retriever_params=retriever_runtime_params,
        retrieve_k=retrieve_k,
        metric_ks=metric_ks,
        split="strict_holdout",
        fold=None,
    )
    topic_metrics_frames.append(holdout_topic_metrics)

    aggregated_metrics = aggregate_metric_summary(fold_metrics, holdout_metrics, metric_ks)
    primary_metric_name = f"recall@{args.top_k}"

    summary_metrics: dict[str, object] = {
        "experiment_name": args.experiment_name,
        "retriever_name": args.retriever,
        "retriever_params": retriever_params,
        "metric_name": primary_metric_name,
        "metric_value": aggregated_metrics[primary_metric_name]["strict_holdout"],
        "strict_cv_mean": aggregated_metrics[primary_metric_name]["strict_cv_mean"],
        "strict_cv_std": aggregated_metrics[primary_metric_name]["strict_cv_std"],
        "strict_holdout": aggregated_metrics[primary_metric_name]["strict_holdout"],
        "num_documents": int(len(documents)),
        "num_train_queries": int(len(train)),
        "num_dev_queries": int(len(dev)),
        "num_holdout_queries": int(len(holdout)),
        "top_k": int(args.top_k),
        "retrieve_k": int(retrieve_k),
        "extra_metric_ks": [metric_k for metric_k in metric_ks if metric_k != args.top_k],
        "validation_splits": str(resolve_path(args.validation_splits)),
        "fold_metrics": fold_metrics,
        "all_metrics": aggregated_metrics,
    }

    predictions_dir = paths.outputs_dir / "predictions"
    metrics_dir = paths.outputs_dir / "metrics"

    cv_predictions_path = predictions_dir / f"{args.experiment_name}_strict_cv_predictions.csv"
    holdout_predictions_path = predictions_dir / f"{args.experiment_name}_strict_holdout_predictions.csv"
    metrics_path = metrics_dir / f"{args.experiment_name}_metrics.json"
    topic_metrics_path = metrics_dir / f"{args.experiment_name}_topic_metrics.csv"

    pd.concat(fold_predictions, ignore_index=True).to_csv(cv_predictions_path, index=False)
    holdout_predictions.to_csv(holdout_predictions_path, index=False)
    pd.concat(topic_metrics_frames, ignore_index=True).to_csv(topic_metrics_path, index=False)
    metrics_path.write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved strict CV predictions to {cv_predictions_path}")
    print(f"Saved strict holdout predictions to {holdout_predictions_path}")
    print(f"Saved topic metrics to {topic_metrics_path}")
    print(f"Saved metrics to {metrics_path}")
    print(
        f"{primary_metric_name} strict_cv_mean={aggregated_metrics[primary_metric_name]['strict_cv_mean']:.4f}"
    )
    print(
        f"{primary_metric_name} strict_cv_std={aggregated_metrics[primary_metric_name]['strict_cv_std']:.4f}"
    )
    print(
        f"{primary_metric_name} strict_holdout={aggregated_metrics[primary_metric_name]['strict_holdout']:.4f}"
    )
    for metric_k in metric_ks:
        metric_name = f"recall@{metric_k}"
        if metric_name == primary_metric_name:
            continue
        metric_summary = aggregated_metrics[metric_name]
        print(
            f"{metric_name} strict_cv_mean={metric_summary['strict_cv_mean']:.4f} "
            f"strict_holdout={metric_summary['strict_holdout']:.4f}"
        )

    if args.save_test_predictions:
        retriever = create_retriever(args.retriever, top_k=retrieve_k, **retriever_runtime_params)
        retriever.fit(documents, train_queries=train)
        test_doc_ids, test_scores = retriever.retrieve(test["question"].tolist(), top_k=retrieve_k)
        test_predictions = build_ranked_predictions(test["qid"].tolist(), test_doc_ids, test_scores)
        test_predictions_path = predictions_dir / f"{args.experiment_name}_test_predictions.csv"
        test_predictions.to_csv(test_predictions_path, index=False)
        print(f"Saved test predictions to {test_predictions_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
