from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class ValidationConfig:
    holdout_splits: int = 5
    strict_cv_splits: int = 4
    relaxed_cv_splits: int = 5
    min_topic_count_for_stratify: int = 10
    random_state: int = 42
    rare_topic_label: str = "__rare_topic__"


def _validate_train_columns(train: pd.DataFrame) -> None:
    required_columns = {"qid", "question", "gold_doc_id", "topic"}
    missing = required_columns - set(train.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Train dataframe is missing columns: {missing_columns}")


def collapse_rare_topics(
    topics: pd.Series,
    *,
    min_count: int,
    rare_label: str = "__rare_topic__",
) -> pd.Series:
    counts = topics.value_counts()
    return topics.where(topics.map(counts) >= min_count, rare_label)


def _topic_distribution_score(
    train: pd.DataFrame,
    val_indices: pd.Index,
    *,
    topic_column: str,
    expected_size: float,
) -> float:
    full_distribution = train[topic_column].value_counts(normalize=True)
    val_distribution = train.loc[val_indices, topic_column].value_counts(normalize=True)
    aligned = (
        pd.concat([full_distribution.rename("full"), val_distribution.rename("val")], axis=1)
        .fillna(0.0)
    )
    size_penalty = abs(len(val_indices) - expected_size) / expected_size
    topic_penalty = (aligned["full"] - aligned["val"]).abs().sum()
    return float(size_penalty + topic_penalty)


def choose_holdout_fold(
    train: pd.DataFrame,
    candidate_val_indices: list[pd.Index],
    *,
    topic_column: str,
) -> int:
    expected_size = len(train) / len(candidate_val_indices)
    scored_folds = [
        (
            fold,
            _topic_distribution_score(
                train,
                val_indices,
                topic_column=topic_column,
                expected_size=expected_size,
            ),
        )
        for fold, val_indices in enumerate(candidate_val_indices)
    ]
    scored_folds.sort(key=lambda item: (item[1], item[0]))
    return scored_folds[0][0]


def build_validation_assignments(
    train: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_train_columns(train)
    config = config or ValidationConfig()

    assignments = train[["qid", "question", "gold_doc_id", "topic"]].copy()
    assignments["topic_bucket"] = collapse_rare_topics(
        assignments["topic"],
        min_count=config.min_topic_count_for_stratify,
        rare_label=config.rare_topic_label,
    )

    holdout_splitter = StratifiedGroupKFold(
        n_splits=config.holdout_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    holdout_candidates = [
        assignments.index[val_idx]
        for _, val_idx in holdout_splitter.split(
            assignments,
            y=assignments["topic_bucket"],
            groups=assignments["gold_doc_id"],
        )
    ]
    chosen_holdout_fold = choose_holdout_fold(
        assignments,
        holdout_candidates,
        topic_column="topic",
    )
    holdout_indices = holdout_candidates[chosen_holdout_fold]

    assignments["strict_holdout_role"] = "dev"
    assignments.loc[holdout_indices, "strict_holdout_role"] = "holdout"

    dev_assignments = assignments[assignments["strict_holdout_role"] == "dev"].copy()
    dev_assignments["strict_topic_bucket"] = collapse_rare_topics(
        dev_assignments["topic"],
        min_count=config.min_topic_count_for_stratify,
        rare_label=config.rare_topic_label,
    )
    strict_splitter = StratifiedGroupKFold(
        n_splits=config.strict_cv_splits,
        shuffle=True,
        random_state=config.random_state + 1,
    )
    strict_cv_fold = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    for fold, (_, val_idx) in enumerate(
        strict_splitter.split(
            dev_assignments,
            y=dev_assignments["strict_topic_bucket"],
            groups=dev_assignments["gold_doc_id"],
        )
    ):
        strict_indices = dev_assignments.index[val_idx]
        strict_cv_fold.loc[strict_indices] = fold
    assignments["strict_cv_fold"] = strict_cv_fold

    relaxed_splitter = StratifiedKFold(
        n_splits=config.relaxed_cv_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    relaxed_cv_fold = pd.Series(index=assignments.index, dtype="Int64")
    for fold, (_, val_idx) in enumerate(
        relaxed_splitter.split(assignments, y=assignments["topic_bucket"])
    ):
        relaxed_indices = assignments.index[val_idx]
        relaxed_cv_fold.loc[relaxed_indices] = fold
    assignments["relaxed_cv_fold"] = relaxed_cv_fold.astype("Int64")

    metadata = {
        "holdout_fold": int(chosen_holdout_fold),
        "holdout_splits": int(config.holdout_splits),
        "strict_cv_splits": int(config.strict_cv_splits),
        "relaxed_cv_splits": int(config.relaxed_cv_splits),
        "min_topic_count_for_stratify": int(config.min_topic_count_for_stratify),
        "random_state": int(config.random_state),
    }
    return assignments, metadata


def summarize_holdout(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby("strict_holdout_role")
        .agg(
            questions=("qid", "count"),
            gold_docs=("gold_doc_id", "nunique"),
            topics=("topic", "nunique"),
        )
        .reset_index()
        .sort_values("strict_holdout_role")
    )


def summarize_folds(assignments: pd.DataFrame, fold_column: str) -> pd.DataFrame:
    subset = assignments.dropna(subset=[fold_column]).copy()
    subset[fold_column] = subset[fold_column].astype(int)
    return (
        subset.groupby(fold_column)
        .agg(
            questions=("qid", "count"),
            gold_docs=("gold_doc_id", "nunique"),
            topics=("topic", "nunique"),
        )
        .reset_index()
        .sort_values(fold_column)
    )


def summarize_relaxed_doc_leakage(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold in sorted(assignments["relaxed_cv_fold"].dropna().astype(int).unique()):
        is_val = assignments["relaxed_cv_fold"] == fold
        val_doc_ids = set(assignments.loc[is_val, "gold_doc_id"])
        train_doc_ids = set(assignments.loc[~is_val, "gold_doc_id"])
        overlap = len(val_doc_ids & train_doc_ids)
        rows.append(
            {
                "relaxed_cv_fold": fold,
                "val_gold_docs": len(val_doc_ids),
                "train_gold_docs": len(train_doc_ids),
                "overlap_gold_docs": overlap,
                "share_val_docs_seen_in_train": overlap / len(val_doc_ids) if val_doc_ids else 0.0,
            }
        )
    return pd.DataFrame(rows)
