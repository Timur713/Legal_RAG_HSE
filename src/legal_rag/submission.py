from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .evaluation import deduplicate_top_k


def create_submission(
    predictions: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    required_columns = {"qid", "doc_id"}
    missing = required_columns - set(predictions.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Predictions are missing columns: {missing_columns}")

    working = predictions.copy()
    if "rank" not in working.columns:
        working["rank"] = working.groupby("qid").cumcount() + 1

    working["qid"] = working["qid"].astype(str)
    working["doc_id"] = working["doc_id"].astype(str)
    working = working.sort_values(["qid", "rank"], kind="stable")

    rows: list[dict[str, str]] = []
    for qid, group in working.groupby("qid", sort=False):
        top_doc_ids = deduplicate_top_k(group["doc_id"].tolist(), k=top_k)
        rows.extend({"qid": qid, "doc_id": doc_id} for doc_id in top_doc_ids)

    return pd.DataFrame(rows, columns=["qid", "doc_id"])


def validate_submission(
    submission: pd.DataFrame,
    expected_qids: Iterable[object] | None = None,
    valid_doc_ids: Iterable[object] | None = None,
    max_docs_per_qid: int = 5,
) -> None:
    if list(submission.columns) != ["qid", "doc_id"]:
        raise ValueError("Submission must have exactly two columns: qid, doc_id")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing values")
    if submission.duplicated(["qid", "doc_id"]).any():
        raise ValueError("Submission contains duplicate (qid, doc_id) pairs")

    counts = submission.groupby("qid").size()
    if not counts.empty and int(counts.max()) > max_docs_per_qid:
        raise ValueError(f"Submission has more than {max_docs_per_qid} rows for some qid")

    if expected_qids is not None:
        expected = {str(qid) for qid in expected_qids}
        actual = set(submission["qid"].astype(str))
        if expected != actual:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Submission qid mismatch. Missing: {missing}, extra: {extra}")

    if valid_doc_ids is not None:
        valid = {str(doc_id) for doc_id in valid_doc_ids}
        invalid_rows = submission.loc[~submission["doc_id"].astype(str).isin(valid)]
        if not invalid_rows.empty:
            invalid_doc_ids = invalid_rows["doc_id"].astype(str).unique().tolist()
            raise ValueError(f"Submission contains unknown doc_id values: {invalid_doc_ids}")


def save_submission(submission: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    return path
