from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import KFold, StratifiedGroupKFold, StratifiedKFold


@dataclass(frozen=True)
class ValidationConfig:
    holdout_splits: int = 5
    strict_cv_splits: int = 4
    relaxed_cv_splits: int = 5
    min_topic_count_for_stratify: int = 10
    min_topic_doc_count_multiplier_for_stratify: int = 1
    random_state: int = 42
    rare_topic_label: str = "__rare_topic__"


def _validate_train_columns(train: pd.DataFrame) -> None:
    required_columns = {"qid", "question", "gold_doc_id", "topic"}
    missing = required_columns - set(train.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Train dataframe is missing columns: {missing_columns}")


def _topic_stats(topics: pd.Series, gold_doc_ids: pd.Series) -> pd.DataFrame:
    stats = (
        pd.DataFrame({"topic": topics, "gold_doc_id": gold_doc_ids})
        .groupby("topic")
        .agg(
            question_count=("topic", "size"),
            unique_gold_doc_count=("gold_doc_id", "nunique"),
        )
    )
    return stats.sort_index()


def _topic_doc_threshold(*, n_splits: int, multiplier: int) -> int:
    return max(1, int(n_splits) * int(multiplier))


def collapse_rare_topics(
    topics: pd.Series,
    gold_doc_ids: pd.Series,
    *,
    min_question_count: int,
    min_unique_gold_doc_count: int,
    rare_label: str = "__rare_topic__",
) -> tuple[pd.Series, pd.DataFrame]:
    stats = _topic_stats(topics, gold_doc_ids)
    rare_mask = (stats["question_count"] < min_question_count) | (
        stats["unique_gold_doc_count"] < min_unique_gold_doc_count
    )
    rare_topics = stats.index[rare_mask]
    collapsed = topics.where(~topics.isin(rare_topics), rare_label)
    stats = stats.assign(
        is_rare=rare_mask,
        rare_reason=rare_mask.map(
            {
                False: "",
                True: (
                    f"question_count < {min_question_count}"
                    f" or unique_gold_doc_count < {min_unique_gold_doc_count}"
                ),
            }
        ),
    )
    return collapsed, stats.reset_index()


def _distribution_alignment(
    train: pd.DataFrame,
    val_indices: pd.Index,
    *,
    topic_column: str,
) -> pd.DataFrame:
    full_distribution = train[topic_column].value_counts(normalize=True)
    val_distribution = train.loc[val_indices, topic_column].value_counts(normalize=True)
    return (
        pd.concat([full_distribution.rename("full_share"), val_distribution.rename("val_share")], axis=1)
        .fillna(0.0)
        .sort_index()
    )


def _holdout_candidate_score(
    train: pd.DataFrame,
    val_indices: pd.Index,
    *,
    topic_column: str,
    rare_topics: set[str],
    expected_questions: float,
    expected_gold_docs: float,
    expected_questions_per_doc: float,
) -> dict[str, float | int | str]:
    aligned = _distribution_alignment(train, val_indices, topic_column=topic_column)
    val = train.loc[val_indices]
    val_gold_docs = val["gold_doc_id"].nunique()
    val_questions = len(val)
    val_questions_per_doc = val_questions / val_gold_docs if val_gold_docs else 0.0
    present_topics = set(val["topic"].unique())
    missing_topics = sorted(set(train["topic"].unique()) - present_topics)
    missing_rare_topics = sorted(rare_topics - present_topics)

    size_penalty = abs(val_questions - expected_questions) / expected_questions if expected_questions else 0.0
    gold_doc_penalty = (
        abs(val_gold_docs - expected_gold_docs) / expected_gold_docs if expected_gold_docs else 0.0
    )
    qpd_penalty = (
        abs(val_questions_per_doc - expected_questions_per_doc) / expected_questions_per_doc
        if expected_questions_per_doc
        else 0.0
    )
    topic_penalty = (aligned["full_share"] - aligned["val_share"]).abs().sum()
    missing_topic_share = len(missing_topics) / train["topic"].nunique() if train["topic"].nunique() else 0.0
    missing_rare_topic_share = len(missing_rare_topics) / len(rare_topics) if rare_topics else 0.0
    total_score = (
        size_penalty
        + topic_penalty
        + gold_doc_penalty
        + qpd_penalty
        + missing_topic_share
        + missing_rare_topic_share
    )

    return {
        "questions": int(val_questions),
        "gold_docs": int(val_gold_docs),
        "questions_per_doc": float(val_questions_per_doc),
        "topic_distribution_penalty": float(topic_penalty),
        "size_penalty": float(size_penalty),
        "gold_doc_penalty": float(gold_doc_penalty),
        "questions_per_doc_penalty": float(qpd_penalty),
        "missing_topics": int(len(missing_topics)),
        "missing_topics_list": ", ".join(missing_topics),
        "missing_rare_topics": int(len(missing_rare_topics)),
        "missing_rare_topics_list": ", ".join(missing_rare_topics),
        "missing_topic_share": float(missing_topic_share),
        "missing_rare_topic_share": float(missing_rare_topic_share),
        "total_score": float(total_score),
    }


def choose_holdout_fold(
    train: pd.DataFrame,
    candidate_val_indices: list[pd.Index],
    *,
    topic_column: str,
    rare_topics: set[str],
) -> tuple[int, pd.DataFrame]:
    expected_questions = len(train) / len(candidate_val_indices)
    expected_gold_docs = train["gold_doc_id"].nunique() / len(candidate_val_indices)
    expected_questions_per_doc = len(train) / train["gold_doc_id"].nunique()
    scored_folds: list[dict[str, float | int | str]] = []
    for fold, val_indices in enumerate(candidate_val_indices):
        row = _holdout_candidate_score(
            train,
            val_indices,
            topic_column=topic_column,
            rare_topics=rare_topics,
            expected_questions=expected_questions,
            expected_gold_docs=expected_gold_docs,
            expected_questions_per_doc=expected_questions_per_doc,
        )
        row["fold"] = fold
        scored_folds.append(row)

    scored = pd.DataFrame(scored_folds).sort_values(["total_score", "fold"]).reset_index(drop=True)
    return int(scored.iloc[0]["fold"]), scored


def build_validation_assignments(
    train: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float | list[str] | dict[str, int]]]:
    _validate_train_columns(train)
    config = config or ValidationConfig()

    assignments = train[["qid", "question", "gold_doc_id", "topic"]].copy()

    holdout_topic_bucket, holdout_topic_stats = collapse_rare_topics(
        assignments["topic"],
        assignments["gold_doc_id"],
        min_question_count=config.min_topic_count_for_stratify,
        min_unique_gold_doc_count=_topic_doc_threshold(
            n_splits=config.holdout_splits,
            multiplier=config.min_topic_doc_count_multiplier_for_stratify,
        ),
        rare_label=config.rare_topic_label,
    )
    assignments["topic_bucket"] = holdout_topic_bucket

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
    holdout_rare_topics = set(
        holdout_topic_stats.loc[holdout_topic_stats["is_rare"], "topic"].astype(str).tolist()
    )
    chosen_holdout_fold, holdout_candidate_scores = choose_holdout_fold(
        assignments,
        holdout_candidates,
        topic_column="topic",
        rare_topics=holdout_rare_topics,
    )
    holdout_indices = holdout_candidates[chosen_holdout_fold]

    assignments["strict_holdout_role"] = "dev"
    assignments.loc[holdout_indices, "strict_holdout_role"] = "holdout"

    dev_assignments = assignments[assignments["strict_holdout_role"] == "dev"].copy()
    strict_topic_bucket, strict_topic_stats = collapse_rare_topics(
        dev_assignments["topic"],
        dev_assignments["gold_doc_id"],
        min_question_count=config.min_topic_count_for_stratify,
        min_unique_gold_doc_count=_topic_doc_threshold(
            n_splits=config.strict_cv_splits,
            multiplier=config.min_topic_doc_count_multiplier_for_stratify,
        ),
        rare_label=config.rare_topic_label,
    )
    dev_assignments["strict_topic_bucket"] = strict_topic_bucket

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

    relaxed_topic_bucket, _ = collapse_rare_topics(
        dev_assignments["topic"],
        dev_assignments["gold_doc_id"],
        min_question_count=config.min_topic_count_for_stratify,
        min_unique_gold_doc_count=1,
        rare_label=config.rare_topic_label,
    )
    relaxed_label_counts = relaxed_topic_bucket.value_counts()
    relaxed_uses_stratified = bool(
        not relaxed_label_counts.empty and relaxed_label_counts.min() >= config.relaxed_cv_splits
    )
    relaxed_splitter = (
        StratifiedKFold(
            n_splits=config.relaxed_cv_splits,
            shuffle=True,
            random_state=config.random_state,
        )
        if relaxed_uses_stratified
        else KFold(
            n_splits=config.relaxed_cv_splits,
            shuffle=True,
            random_state=config.random_state,
        )
    )
    relaxed_cv_fold = pd.Series(pd.NA, index=assignments.index, dtype="Int64")
    relaxed_split_args = {"X": dev_assignments}
    if relaxed_uses_stratified:
        relaxed_split_args["y"] = relaxed_topic_bucket
    for fold, (_, val_idx) in enumerate(relaxed_splitter.split(**relaxed_split_args)):
        relaxed_indices = dev_assignments.index[val_idx]
        relaxed_cv_fold.loc[relaxed_indices] = fold
    assignments["relaxed_cv_fold"] = relaxed_cv_fold

    multi_topic_counts = train.groupby("gold_doc_id")["topic"].nunique()
    metadata = {
        "holdout_fold": int(chosen_holdout_fold),
        "holdout_splits": int(config.holdout_splits),
        "strict_cv_splits": int(config.strict_cv_splits),
        "relaxed_cv_splits": int(config.relaxed_cv_splits),
        "min_topic_count_for_stratify": int(config.min_topic_count_for_stratify),
        "min_topic_doc_count_multiplier_for_stratify": int(
            config.min_topic_doc_count_multiplier_for_stratify
        ),
        "holdout_min_unique_gold_doc_count_for_stratify": int(
            _topic_doc_threshold(
                n_splits=config.holdout_splits,
                multiplier=config.min_topic_doc_count_multiplier_for_stratify,
            )
        ),
        "strict_min_unique_gold_doc_count_for_stratify": int(
            _topic_doc_threshold(
                n_splits=config.strict_cv_splits,
                multiplier=config.min_topic_doc_count_multiplier_for_stratify,
            )
        ),
        "relaxed_cv_strategy": "stratified" if relaxed_uses_stratified else "kfold_fallback",
        "random_state": int(config.random_state),
        "topics_total": int(train["topic"].nunique()),
        "gold_docs_total": int(train["gold_doc_id"].nunique()),
        "multi_topic_gold_docs": int(multi_topic_counts.gt(1).sum()),
        "max_topics_per_gold_doc": int(multi_topic_counts.max()),
        "rare_topics_for_holdout": sorted(holdout_rare_topics),
        "rare_topics_for_strict_cv": sorted(
            strict_topic_stats.loc[strict_topic_stats["is_rare"], "topic"].astype(str).tolist()
        ),
        "rare_topic_counts": {
            "holdout": int(holdout_topic_stats["is_rare"].sum()),
            "strict_cv_dev": int(strict_topic_stats["is_rare"].sum()),
        },
        "holdout_candidate_scores": holdout_candidate_scores.round(6).to_dict(orient="records"),
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


def summarize_topic_diagnostics(
    assignments: pd.DataFrame,
    *,
    fold_column: str,
) -> dict[str, pd.DataFrame]:
    subset = assignments.dropna(subset=[fold_column]).copy()
    subset[fold_column] = subset[fold_column].astype(int)

    question_distribution = (
        subset.pivot_table(index="topic", columns=fold_column, values="qid", aggfunc="count", fill_value=0)
        .reset_index()
        .sort_values("topic")
    )
    gold_doc_distribution = (
        subset.pivot_table(
            index="topic",
            columns=fold_column,
            values="gold_doc_id",
            aggfunc=pd.Series.nunique,
            fill_value=0,
        )
        .reset_index()
        .sort_values("topic")
    )

    full_distribution = assignments["topic"].value_counts(normalize=True)
    deviation_rows: list[dict[str, float | int | str]] = []
    missing_rows: list[dict[str, int | str]] = []
    for fold in sorted(subset[fold_column].unique()):
        fold_subset = subset[subset[fold_column] == fold]
        fold_distribution = fold_subset["topic"].value_counts(normalize=True)
        aligned = (
            pd.concat([full_distribution.rename("full_share"), fold_distribution.rename("fold_share")], axis=1)
            .fillna(0.0)
            .reset_index()
            .rename(columns={"index": "topic"})
        )
        aligned["fold"] = int(fold)
        aligned["abs_deviation"] = (aligned["full_share"] - aligned["fold_share"]).abs()
        deviation_rows.extend(aligned.to_dict(orient="records"))

        missing_topics = sorted(set(assignments["topic"].unique()) - set(fold_subset["topic"].unique()))
        missing_rows.append(
            {
                "fold": int(fold),
                "missing_topics": int(len(missing_topics)),
                "missing_topics_list": ", ".join(missing_topics),
            }
        )

    deviation_df = pd.DataFrame(deviation_rows).sort_values(
        ["abs_deviation", "fold", "topic"], ascending=[False, True, True]
    )
    missing_df = pd.DataFrame(missing_rows).sort_values("fold")

    return {
        "topic_question_distribution": question_distribution,
        "topic_gold_doc_distribution": gold_doc_distribution,
        "missing_topics": missing_df,
        "largest_topic_deviations": deviation_df,
    }


def summarize_topic_stats(
    assignments: pd.DataFrame,
    *,
    min_question_count: int,
    min_unique_gold_doc_count: int,
) -> pd.DataFrame:
    stats = _topic_stats(assignments["topic"], assignments["gold_doc_id"]).reset_index()
    stats["is_rare"] = (stats["question_count"] < min_question_count) | (
        stats["unique_gold_doc_count"] < min_unique_gold_doc_count
    )
    return stats.sort_values(["is_rare", "unique_gold_doc_count", "question_count", "topic"], ascending=[False, True, True, True])


def summarize_relaxed_doc_leakage(assignments: pd.DataFrame) -> pd.DataFrame:
    subset = assignments.dropna(subset=["relaxed_cv_fold"]).copy()
    subset["relaxed_cv_fold"] = subset["relaxed_cv_fold"].astype(int)
    rows: list[dict[str, float | int]] = []
    for fold in sorted(subset["relaxed_cv_fold"].unique()):
        is_val = subset["relaxed_cv_fold"] == fold
        val_doc_ids = set(subset.loc[is_val, "gold_doc_id"])
        train_doc_ids = set(subset.loc[~is_val, "gold_doc_id"])
        overlap = len(val_doc_ids & train_doc_ids)
        rows.append(
            {
                "relaxed_cv_fold": int(fold),
                "val_gold_docs": len(val_doc_ids),
                "train_gold_docs": len(train_doc_ids),
                "overlap_gold_docs": overlap,
                "share_val_docs_seen_in_train": overlap / len(val_doc_ids) if val_doc_ids else 0.0,
            }
        )
    return pd.DataFrame(rows)
