from __future__ import annotations

import argparse
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
)
from legal_rag.submission import create_submission, save_submission, validate_submission  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a submission.csv from saved predictions.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--experiment-name",
        default="baseline",
        help="Prefix used by scripts/run_baseline.py when it saved predictions.",
    )
    parser.add_argument(
        "--predictions-file",
        default=None,
        help="Optional explicit path to a predictions CSV. Defaults to outputs/predictions/<experiment>_test_predictions.csv.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of documents per question in the final submission.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)
    ensure_output_dirs(paths)

    documents = load_documents(paths)
    test = load_test(paths)

    default_predictions_path = paths.outputs_dir / "predictions" / f"{args.experiment_name}_test_predictions.csv"
    predictions_path = Path(args.predictions_file) if args.predictions_file else default_predictions_path
    if not predictions_path.is_absolute():
        predictions_path = PROJECT_ROOT / predictions_path

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Predictions file does not exist: {predictions_path}. Run scripts/run_baseline.py first or pass --predictions-file."
        )

    predictions = pd.read_csv(predictions_path)
    submission = create_submission(predictions, top_k=args.top_k)
    validate_submission(
        submission,
        expected_qids=test["qid"].tolist(),
        valid_doc_ids=documents["doc_id"].tolist(),
        max_docs_per_qid=args.top_k,
    )

    output_path = paths.outputs_dir / "submissions" / f"{args.experiment_name}_submission.csv"
    save_submission(submission, output_path)

    print(f"Saved submission to {output_path}")
    print(f"rows={len(submission)} qids={submission['qid'].nunique()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
