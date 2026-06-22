from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.data import ensure_output_dirs, load_paths_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict-CV sweep over chunking configurations."
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--validation-splits",
        default="data/processed/validation/validation_splits.csv",
        help="Path to validation_splits.csv created by scripts/make_validation_splits.py.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default="chunking_sweep",
        help="Prefix for experiment names and summary artifacts.",
    )
    parser.add_argument(
        "--retriever",
        default="chunked_bm25",
        choices=("chunked_tfidf", "chunked_bm25", "hybrid_rrf", "cross_encoder_rerank"),
        help="Retriever family to evaluate with different chunking configurations.",
    )
    parser.add_argument(
        "--chunk-configs",
        nargs="*",
        default=("900:450", "1200:600", "1600:800", "2000:1000"),
        help="Chunk configurations in SIZE:STRIDE format.",
    )
    parser.add_argument(
        "--chunk-score-aggregation",
        default="logsumexp",
        help="Chunk-score aggregation mode for chunked retrievers.",
    )
    parser.add_argument(
        "--chunk-score-top-k",
        type=int,
        default=3,
        help="Top-k parameter used by top-k-based chunk score aggregation modes.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Primary Recall@k metric to optimize and compare.",
    )
    parser.add_argument(
        "--extra-metric-k",
        type=int,
        nargs="*",
        default=(20,),
        help="Optional extra Recall@k values to compute.",
    )
    parser.add_argument(
        "--use-lemmas",
        action="store_true",
        help="Apply Russian lemmatization with pymorphy3 during tokenization.",
    )
    parser.add_argument(
        "--hybrid-retrievers",
        nargs="*",
        default=("bm25", "chunked_bm25"),
        help="Component retrievers for hybrid_rrf.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion constant for hybrid_rrf.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="How many first-stage candidates to rerank for cross_encoder_rerank.",
    )
    parser.add_argument(
        "--first-stage-retriever",
        default="hybrid_rrf",
        help="First-stage retriever used by cross_encoder_rerank.",
    )
    parser.add_argument(
        "--first-stage-hybrid-retrievers",
        nargs="*",
        default=("bm25", "chunked_bm25"),
        help="Component retrievers for a hybrid first stage inside cross_encoder_rerank.",
    )
    parser.add_argument(
        "--first-stage-rrf-k",
        type=int,
        default=60,
        help="RRF constant for a hybrid first stage inside cross_encoder_rerank.",
    )
    parser.add_argument(
        "--reranker-model-name",
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        help="Cross-encoder model for reranking.",
    )
    parser.add_argument(
        "--reranker-batch-size",
        type=int,
        default=16,
        help="Batch size for cross-encoder inference.",
    )
    parser.add_argument(
        "--reranker-max-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length for the cross-encoder.",
    )
    parser.add_argument(
        "--reranker-device",
        default=None,
        help="Optional device override for cross-encoder, for example cuda or cpu.",
    )
    parser.add_argument(
        "--reranker-chunks-per-doc",
        type=int,
        default=3,
        help="How many top chunks per document to score with the cross-encoder.",
    )
    parser.add_argument(
        "--reranker-combine-mode",
        default="rrf",
        choices=("ce_only", "rrf", "linear"),
        help="How to combine cross-encoder scores with first-stage scores.",
    )
    parser.add_argument(
        "--reranker-ce-weight",
        type=float,
        default=0.7,
        help="Cross-encoder weight for reranker score blending.",
    )
    parser.add_argument(
        "--reranker-sparse-weight",
        type=float,
        default=0.3,
        help="First-stage sparse weight for reranker score blending.",
    )
    parser.add_argument(
        "--reranker-rrf-k",
        type=int,
        default=60,
        help="RRF constant used when --reranker-combine-mode rrf.",
    )
    return parser.parse_args()


def parse_chunk_config(raw_value: str) -> tuple[int, int]:
    try:
        size_raw, stride_raw = raw_value.split(":", maxsplit=1)
        chunk_size = int(size_raw)
        chunk_stride = int(stride_raw)
    except ValueError as error:
        raise ValueError(
            f"Invalid chunk config {raw_value!r}. Use SIZE:STRIDE, for example 1600:800."
        ) from error

    if chunk_size <= 0 or chunk_stride <= 0:
        raise ValueError(f"Chunk config must be positive: {raw_value!r}.")
    if chunk_stride > chunk_size:
        raise ValueError(f"Chunk stride cannot exceed chunk size: {raw_value!r}.")
    return chunk_size, chunk_stride


def build_run_command(
    args: argparse.Namespace,
    *,
    experiment_name: str,
    chunk_size: int,
    chunk_stride: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_baseline_validation.py"),
        "--paths",
        str(args.paths),
        "--validation-splits",
        str(args.validation_splits),
        "--experiment-name",
        experiment_name,
        "--retriever",
        str(args.retriever),
        "--top-k",
        str(args.top_k),
        "--chunk-size",
        str(chunk_size),
        "--chunk-stride",
        str(chunk_stride),
        "--chunk-score-aggregation",
        str(args.chunk_score_aggregation),
        "--chunk-score-top-k",
        str(args.chunk_score_top_k),
    ]

    if args.extra_metric_k:
        command.append("--extra-metric-k")
        command.extend(str(metric_k) for metric_k in args.extra_metric_k)

    if args.use_lemmas:
        command.append("--use-lemmas")

    if args.retriever == "hybrid_rrf":
        command.extend(("--rrf-k", str(args.rrf_k)))
        if args.hybrid_retrievers:
            command.append("--hybrid-retrievers")
            command.extend(str(value) for value in args.hybrid_retrievers)

    if args.retriever == "cross_encoder_rerank":
        command.extend(
            (
                "--rerank-top-k",
                str(args.rerank_top_k),
                "--first-stage-retriever",
                str(args.first_stage_retriever),
                "--first-stage-rrf-k",
                str(args.first_stage_rrf_k),
                "--reranker-model-name",
                str(args.reranker_model_name),
                "--reranker-batch-size",
                str(args.reranker_batch_size),
                "--reranker-max-length",
                str(args.reranker_max_length),
                "--reranker-chunks-per-doc",
                str(args.reranker_chunks_per_doc),
                "--reranker-combine-mode",
                str(args.reranker_combine_mode),
                "--reranker-ce-weight",
                str(args.reranker_ce_weight),
                "--reranker-sparse-weight",
                str(args.reranker_sparse_weight),
                "--reranker-rrf-k",
                str(args.reranker_rrf_k),
            )
        )
        if args.reranker_device:
            command.extend(("--reranker-device", str(args.reranker_device)))
        if args.first_stage_hybrid_retrievers:
            command.append("--first-stage-hybrid-retrievers")
            command.extend(str(value) for value in args.first_stage_hybrid_retrievers)

    return command


def main() -> int:
    args = parse_args()
    if args.chunk_score_top_k <= 0:
        raise ValueError("chunk_score_top_k must be positive.")

    paths = load_paths_config(args.paths)
    ensure_output_dirs(paths)

    chunk_configs = [parse_chunk_config(raw_value) for raw_value in args.chunk_configs]
    rows: list[dict[str, object]] = []

    for chunk_size, chunk_stride in chunk_configs:
        experiment_name = f"{args.experiment_prefix}_{args.retriever}_{chunk_size}_{chunk_stride}"
        command = build_run_command(
            args,
            experiment_name=experiment_name,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
        )
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        metrics_path = paths.outputs_dir / "metrics" / f"{experiment_name}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "experiment_name": experiment_name,
                "retriever_name": metrics["retriever_name"],
                "chunk_size": chunk_size,
                "chunk_stride": chunk_stride,
                "chunk_overlap": chunk_size - chunk_stride,
                "chunk_score_aggregation": args.chunk_score_aggregation,
                "chunk_score_top_k": int(args.chunk_score_top_k),
                "metric_name": metrics["metric_name"],
                "strict_cv_mean": float(metrics["strict_cv_mean"]),
                "strict_cv_std": float(metrics["strict_cv_std"]),
                "strict_holdout": float(metrics["strict_holdout"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["strict_cv_mean", "strict_holdout", "strict_cv_std"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    summary_dir = paths.outputs_dir / "metrics"
    csv_path = summary_dir / f"{args.experiment_prefix}_{args.retriever}_summary.csv"
    json_path = summary_dir / f"{args.experiment_prefix}_{args.retriever}_summary.json"

    summary.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved summary JSON to {json_path}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
