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

from legal_rag.data import load_paths_config, load_train  # noqa: E402
from legal_rag.validation import (  # noqa: E402
    ValidationConfig,
    build_validation_assignments,
    summarize_folds,
    summarize_holdout,
    summarize_topic_diagnostics,
    summarize_topic_stats,
    summarize_relaxed_doc_leakage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fixed local validation splits.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for validation artifacts. Defaults to <processed_data_dir>/validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for fold generation.",
    )
    parser.add_argument(
        "--min-topic-count-for-stratify",
        type=int,
        default=10,
        help="Topics with fewer questions are merged into a rare bucket for splitting.",
    )
    parser.add_argument(
        "--min-topic-doc-count-multiplier-for-stratify",
        type=int,
        default=1,
        help="Topics with fewer than multiplier * n_splits unique gold_doc_id are merged into a rare bucket.",
    )
    return parser.parse_args()


def format_table(dataframe: pd.DataFrame, *, index: bool = False) -> str:
    return f"```text\n{dataframe.to_string(index=index)}\n```"


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)
    train = load_train(paths)

    output_dir = Path(args.output_dir) if args.output_dir else paths.processed_data_dir / "validation"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ValidationConfig(
        random_state=args.random_state,
        min_topic_count_for_stratify=args.min_topic_count_for_stratify,
        min_topic_doc_count_multiplier_for_stratify=args.min_topic_doc_count_multiplier_for_stratify,
    )
    assignments, metadata = build_validation_assignments(train, config=config)

    holdout_summary = summarize_holdout(assignments)
    strict_cv_summary = summarize_folds(assignments, "strict_cv_fold")
    relaxed_cv_summary = summarize_folds(assignments, "relaxed_cv_fold")
    holdout_topic_stats = summarize_topic_stats(
        assignments,
        min_question_count=config.min_topic_count_for_stratify,
        min_unique_gold_doc_count=metadata["holdout_min_unique_gold_doc_count_for_stratify"],
    )
    strict_topic_stats = summarize_topic_stats(
        assignments[assignments["strict_holdout_role"] == "dev"].copy(),
        min_question_count=config.min_topic_count_for_stratify,
        min_unique_gold_doc_count=metadata["strict_min_unique_gold_doc_count_for_stratify"],
    )
    strict_topic_diagnostics = summarize_topic_diagnostics(assignments, fold_column="strict_cv_fold")
    relaxed_topic_diagnostics = summarize_topic_diagnostics(assignments, fold_column="relaxed_cv_fold")
    relaxed_leakage_summary = summarize_relaxed_doc_leakage(assignments).round(3)
    holdout_candidate_scores = pd.DataFrame(metadata["holdout_candidate_scores"]).round(3)
    metadata_table = pd.DataFrame(
        [
            {
                "holdout_fold": metadata["holdout_fold"],
                "holdout_splits": metadata["holdout_splits"],
                "strict_cv_splits": metadata["strict_cv_splits"],
                "relaxed_cv_splits": metadata["relaxed_cv_splits"],
                "relaxed_cv_strategy": metadata["relaxed_cv_strategy"],
                "min_topic_count_for_stratify": metadata["min_topic_count_for_stratify"],
                "min_topic_doc_count_multiplier_for_stratify": metadata[
                    "min_topic_doc_count_multiplier_for_stratify"
                ],
                "holdout_min_unique_gold_doc_count_for_stratify": metadata[
                    "holdout_min_unique_gold_doc_count_for_stratify"
                ],
                "strict_min_unique_gold_doc_count_for_stratify": metadata[
                    "strict_min_unique_gold_doc_count_for_stratify"
                ],
                "random_state": metadata["random_state"],
                "topics_total": metadata["topics_total"],
                "gold_docs_total": metadata["gold_docs_total"],
                "multi_topic_gold_docs": metadata["multi_topic_gold_docs"],
                "max_topics_per_gold_doc": metadata["max_topics_per_gold_doc"],
            }
        ]
    )
    strict_largest_deviations = (
        strict_topic_diagnostics["largest_topic_deviations"]
        .head(20)
        .round({"full_share": 3, "fold_share": 3, "abs_deviation": 3})
    )
    relaxed_largest_deviations = (
        relaxed_topic_diagnostics["largest_topic_deviations"]
        .head(20)
        .round({"full_share": 3, "fold_share": 3, "abs_deviation": 3})
    )

    assignments_path = output_dir / "validation_splits.csv"
    metadata_path = output_dir / "validation_metadata.json"
    protocol_path = output_dir / "validation_protocol.md"

    assignments.to_csv(assignments_path, index=False)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    protocol_lines: list[str] = []
    protocol_lines.append("# Local Validation Protocol")
    protocol_lines.append("")
    protocol_lines.append("## Основная идея")
    protocol_lines.append("")
    protocol_lines.append(
        "- `strict` режим является основным: holdout и CV строятся с группировкой по `gold_doc_id`, "
        "чтобы один и тот же правильный документ не попадал одновременно в train и validation."
    )
    protocol_lines.append(
        "- `relaxed` режим является вспомогательным: обычный стратифицированный CV по вопросам показывает, "
        "как модель ведет себя в более легком сценарии `новый вопрос про уже знакомый документ`."
    )
    protocol_lines.append(
        "- Темы с очень малым числом вопросов или уникальных `gold_doc_id` объединяются в редкую корзину, "
        "потому что для `StratifiedGroupKFold` важна не только частота строк, но и число доступных групп."
    )
    protocol_lines.append("")
    protocol_lines.append("## Ограничения данных")
    protocol_lines.append("")
    protocol_lines.append(
        f"- В train {metadata['multi_topic_gold_docs']} `gold_doc_id` встречаются более чем в одной теме; максимум тем на один документ: {metadata['max_topics_per_gold_doc']}."
    )
    protocol_lines.append(
        "- Поэтому `strict` разбиение является в первую очередь group-valid по `gold_doc_id`, а topic-стратификация в нем неизбежно шумная."
    )
    protocol_lines.append("")
    protocol_lines.append("## Что использовать в работе")
    protocol_lines.append("")
    protocol_lines.append(
        "- Для повседневного отбора моделей использовать только `strict_cv_fold` на строках, где `strict_holdout_role=dev`."
    )
    protocol_lines.append(
        "- `strict_holdout_role=holdout` не трогать в цикле подбора гиперпараметров; использовать только для редких финальных проверок перед внешним сабмитом."
    )
    protocol_lines.append(
        "- `relaxed_cv_fold` смотреть как дополнительный индикатор. Он строится только на `dev`; строки `strict_holdout_role=holdout` имеют `NA` и не должны попадать во вспомогательные CV-эксперименты."
    )
    protocol_lines.append("")
    protocol_lines.append("## Метрики для каждого эксперимента")
    protocol_lines.append("")
    protocol_lines.append("- Основная: mean `Recall@5` по `strict_cv_fold`.")
    protocol_lines.append("- Стабильность: std `Recall@5` по `strict_cv_fold`.")
    protocol_lines.append("- Контроль регрессий: `Recall@5` по темам на strict validation.")
    protocol_lines.append("- Дополнительно для двухэтапных систем: `Recall@20` первого этапа и `Recall@5` после rerank.")
    protocol_lines.append("")
    protocol_lines.append("## Артефакты")
    protocol_lines.append("")
    protocol_lines.append(f"- `validation_splits.csv`: `{assignments_path}`")
    protocol_lines.append(f"- `validation_metadata.json`: `{metadata_path}`")
    protocol_lines.append("")
    protocol_lines.append("## Параметры протокола")
    protocol_lines.append("")
    protocol_lines.append(format_table(metadata_table, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Разбиение Holdout")
    protocol_lines.append("")
    protocol_lines.append(format_table(holdout_summary, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Редкие темы для Holdout")
    protocol_lines.append("")
    protocol_lines.append(format_table(holdout_topic_stats, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Кандидаты Holdout")
    protocol_lines.append("")
    protocol_lines.append(
        "Holdout выбирается не только по размеру и topic distribution, но и по числу `gold_doc_id`, нагрузке вопросов на документ и покрытию редких тем."
    )
    protocol_lines.append("")
    protocol_lines.append(format_table(holdout_candidate_scores, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Разбиение Strict CV")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_cv_summary, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Редкие темы для Strict CV")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_topic_stats, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Strict CV: вопросы по темам и фолдам")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_topic_diagnostics["topic_question_distribution"], index=False))
    protocol_lines.append("")
    protocol_lines.append("## Strict CV: уникальные gold_doc_id по темам и фолдам")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_topic_diagnostics["topic_gold_doc_distribution"], index=False))
    protocol_lines.append("")
    protocol_lines.append("## Strict CV: отсутствующие темы по фолдам")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_topic_diagnostics["missing_topics"], index=False))
    protocol_lines.append("")
    protocol_lines.append("## Strict CV: самые большие отклонения от полной topic distribution")
    protocol_lines.append("")
    protocol_lines.append(format_table(strict_largest_deviations, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Разбиение Relaxed CV")
    protocol_lines.append("")
    protocol_lines.append(format_table(relaxed_cv_summary, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Relaxed CV: отсутствующие темы по фолдам")
    protocol_lines.append("")
    protocol_lines.append(format_table(relaxed_topic_diagnostics["missing_topics"], index=False))
    protocol_lines.append("")
    protocol_lines.append("## Relaxed CV: самые большие отклонения от полной topic distribution")
    protocol_lines.append("")
    protocol_lines.append(format_table(relaxed_largest_deviations, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Leakage в Relaxed CV")
    protocol_lines.append("")
    protocol_lines.append(
        "Здесь видно, почему `relaxed` нельзя использовать как единственный офлайн-сигнал: почти все "
        "валидационные `gold_doc_id` уже встречаются в train-части того же фолда."
    )
    protocol_lines.append("")
    protocol_lines.append(format_table(relaxed_leakage_summary, index=False))
    protocol_lines.append("")
    protocol_lines.append("## Решение о внешнем сабмите")
    protocol_lines.append("")
    protocol_lines.append(
        "- Тратить один из 5 сабмитов только если эксперимент улучшил mean `Recall@5` на `strict_cv` и не дал явной деградации по ключевым темам."
    )
    protocol_lines.append(
        "- Если прирост маленький, ориентироваться на несколько запусков с разными seeds и сравнивать не один фолд, а среднее и разброс."
    )
    protocol_lines.append("")

    protocol_path.write_text("\n".join(protocol_lines), encoding="utf-8")

    print(f"Saved validation splits to {assignments_path}")
    print(f"Saved validation metadata to {metadata_path}")
    print(f"Saved validation protocol to {protocol_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
