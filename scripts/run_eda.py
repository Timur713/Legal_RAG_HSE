from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.baseline import TfidfRetriever  # noqa: E402
from legal_rag.data import load_documents, load_paths_config, load_test, load_train  # noqa: E402
from legal_rag.preprocessing import normalize_text, tokenize  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval-oriented EDA and save a markdown report.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for EDA artifacts. Defaults to <outputs_dir>/eda.",
    )
    return parser.parse_args()


def doc_kind(text: object) -> str:
    raw_text = str(text)
    header = raw_text[:800].upper().replace(" ", "").replace("\n", "")
    if "АПЕЛЛЯЦИОННОЕОПРЕДЕЛЕНИЕ" in header:
        return "апелляционное определение"
    if "КАССАЦИОН" in raw_text[:1500].upper():
        return "кассационное определение"
    if "РЕШЕНИЕ" in header[:300]:
        return "решение"
    return "определение"


def doc_instance(text: object) -> str:
    header = str(text)[:1500].upper()
    if "КАССАЦИОН" in header:
        return "кассация"
    if "АПЕЛЛЯЦИОН" in header:
        return "апелляция"
    return "первая инстанция"


def describe_series(series: pd.Series) -> dict[str, float]:
    stats = series.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    result: dict[str, float] = {}
    for key, value in stats.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            result[str(key)] = round(float(value), 3)
    return result


def overlap_ratio(left_text: object, right_text: object) -> float:
    left_tokens = set(tokenize(left_text))
    right_tokens = set(tokenize(right_text))
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def format_table(dataframe: pd.DataFrame, index: bool = True) -> str:
    return f"```text\n{dataframe.to_string(index=index)}\n```"


def format_mapping(mapping: dict[str, float | int]) -> str:
    return f"```text\n{pd.Series(mapping).to_string()}\n```"


def bullet(text: str) -> str:
    return f"- {text}"


def safe_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)

    output_dir = Path(args.output_dir) if args.output_dir else paths.outputs_dir / "eda"
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = load_documents(paths).copy()
    train = load_train(paths).copy()
    test = load_test(paths).copy()

    raw_shapes = {
        "documents": documents.shape,
        "train": train.shape,
        "test": test.shape,
    }

    documents["text_norm"] = documents["text"].map(normalize_text)
    documents["doc_kind"] = documents["text"].map(doc_kind)
    documents["instance"] = documents["text"].map(doc_instance)
    documents["doc_chars"] = documents["text"].str.len()
    documents["doc_tokens"] = documents["text"].map(lambda value: len(tokenize(value)))

    train["q_chars"] = train["question"].str.len()
    train["q_tokens"] = train["question"].map(lambda value: len(tokenize(value)))
    test["q_chars"] = test["question"].str.len()
    test["q_tokens"] = test["question"].map(lambda value: len(tokenize(value)))
    train["ideal_answer_chars"] = train["ideal_answer"].str.len()
    train["evidence_chars"] = train["gold_evidence_text"].str.len()
    train["evidence_tokens"] = train["gold_evidence_text"].map(lambda value: len(tokenize(value)))
    train["evidence_start"] = train["gold_evidence_char_start"].astype(int)
    train["evidence_end"] = train["gold_evidence_char_end"].astype(int)
    train["evidence_span"] = train["evidence_end"] - train["evidence_start"]

    doc_text_by_id = documents.set_index("doc_id")["text"].to_dict()
    evidence_offset_valid = 0
    for row in train.itertuples(index=False):
        text = doc_text_by_id[row.gold_doc_id]
        fragment = text[int(row.evidence_start) : int(row.evidence_end)]
        evidence_offset_valid += int(fragment == row.gold_evidence_text)

    train = train.merge(
        documents[["doc_id", "doc_kind", "instance", "doc_chars", "doc_tokens"]],
        left_on="gold_doc_id",
        right_on="doc_id",
        how="left",
    )
    train["evidence_pos_ratio"] = np.where(
        train["doc_chars"] > 0,
        train["evidence_start"] / train["doc_chars"],
        np.nan,
    )
    train["question_vs_evidence_overlap"] = [
        overlap_ratio(question, evidence)
        for question, evidence in zip(train["question"], train["gold_evidence_text"])
    ]
    train["question_vs_doc_overlap"] = [
        overlap_ratio(question, doc_text_by_id[gold_doc_id])
        for question, gold_doc_id in zip(train["question"], train["gold_doc_id"])
    ]

    duplicate_text_groups = (
        documents.groupby("text_norm")
        .agg(doc_count=("doc_id", "size"), doc_ids=("doc_id", lambda series: list(series)))
        .reset_index()
    )
    duplicate_text_groups = duplicate_text_groups[duplicate_text_groups["doc_count"] > 1].sort_values(
        "doc_count",
        ascending=False,
    )

    unique_gold_docs = train["gold_doc_id"].nunique()
    gold_doc_frequency = train["gold_doc_id"].value_counts()

    retriever = TfidfRetriever(top_k=5).fit(documents[["doc_id", "text"]])
    ranked_doc_ids, _ = retriever.retrieve(train["question"].tolist(), top_k=5)
    train["hit_at_5"] = [gold_doc_id in prediction for gold_doc_id, prediction in zip(train["gold_doc_id"], ranked_doc_ids)]
    train["rank"] = [
        next((rank for rank, doc_id in enumerate(prediction, start=1) if doc_id == gold_doc_id), math.nan)
        for gold_doc_id, prediction in zip(train["gold_doc_id"], ranked_doc_ids)
    ]
    train["top1_pred"] = [prediction[0] for prediction in ranked_doc_ids]

    baseline_recall_at_5 = float(train["hit_at_5"].mean())
    baseline_top1_accuracy = float((train["rank"] == 1).mean())

    baseline_by_topic = (
        train.groupby("topic")
        .agg(questions=("qid", "count"), recall_at_5=("hit_at_5", "mean"))
        .sort_values(["recall_at_5", "questions"])
    )
    baseline_by_doc_kind = (
        train.groupby("doc_kind")
        .agg(questions=("qid", "count"), recall_at_5=("hit_at_5", "mean"))
        .sort_values("recall_at_5")
    )
    baseline_by_instance = (
        train.groupby("instance")
        .agg(questions=("qid", "count"), recall_at_5=("hit_at_5", "mean"))
        .sort_values("recall_at_5")
    )

    train_vocab = {token for question in train["question"] for token in tokenize(question)}
    test_vocab = {token for question in test["question"] for token in tokenize(question)}
    test_unique_tokens = [token for question in test["question"] for token in set(tokenize(question))]
    test_token_coverage = (
        sum(token in train_vocab for token in test_unique_tokens) / len(test_unique_tokens)
        if test_unique_tokens
        else 0.0
    )

    question_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    train_question_matrix = question_vectorizer.fit_transform(train["question"].map(normalize_text))
    test_question_matrix = question_vectorizer.transform(test["question"].map(normalize_text))
    test_to_train_similarity = linear_kernel(test_question_matrix, train_question_matrix)
    max_train_question_similarity = test_to_train_similarity.max(axis=1)

    doc_vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    doc_matrix = doc_vectorizer.fit_transform(documents["text_norm"])
    doc_similarity = linear_kernel(doc_matrix, doc_matrix)
    np.fill_diagonal(doc_similarity, -1.0)
    max_doc_nn_similarity = doc_similarity.max(axis=1)

    hard_gold_docs = (
        train.groupby("gold_doc_id")
        .agg(questions=("qid", "count"), recall_at_5=("hit_at_5", "mean"))
        .sort_values(["recall_at_5", "questions"])
    )
    frequent_hard_gold_docs = hard_gold_docs[hard_gold_docs["questions"] >= 8].head(12)

    dataset_shapes = pd.DataFrame(
        [
            {"dataset": "documents", "rows": raw_shapes["documents"][0], "columns": raw_shapes["documents"][1]},
            {"dataset": "train", "rows": raw_shapes["train"][0], "columns": raw_shapes["train"][1]},
            {"dataset": "test", "rows": raw_shapes["test"][0], "columns": raw_shapes["test"][1]},
        ]
    )
    doc_kind_counts = documents["doc_kind"].value_counts().rename_axis("doc_kind").reset_index(name="count")
    instance_counts = documents["instance"].value_counts().rename_axis("instance").reset_index(name="count")
    top_topics = train["topic"].value_counts().rename_axis("topic").reset_index(name="questions")

    gold_coverage = pd.DataFrame(
        [
            {"metric": "unique_gold_docs", "value": unique_gold_docs},
            {"metric": "corpus_docs", "value": len(documents)},
            {"metric": "unused_docs_in_train", "value": len(documents) - unique_gold_docs},
            {"metric": "gold_docs_used_once", "value": int((gold_doc_frequency == 1).sum())},
            {"metric": "gold_docs_used_5plus", "value": int((gold_doc_frequency >= 5).sum())},
        ]
    )

    report_lines: list[str] = []
    report_lines.append("# Retrieval-Oriented EDA")
    report_lines.append("")
    report_lines.append("## Контекст")
    report_lines.append("")
    report_lines.append(
        "Изучен локальный референс-бейзлайн [`notebooks/baseline_reference.ipynb`](../../notebooks/baseline_reference.ipynb) "
        "и текущая реализация `src/legal_rag/baseline.py`: в проекте используется простой retrieval на TF-IDF "
        "с regex-токенизацией, ручным списком стоп-слов и косинусной близостью по полным документам."
    )
    report_lines.append("")
    report_lines.append("## 1. Размеры данных")
    report_lines.append("")
    report_lines.append(format_table(dataset_shapes, index=False))
    report_lines.append("")
    report_lines.append("## 2. Корпус документов")
    report_lines.append("")
    report_lines.append("Типы актов:")
    report_lines.append(format_table(doc_kind_counts, index=False))
    report_lines.append("")
    report_lines.append("Инстанции:")
    report_lines.append(format_table(instance_counts, index=False))
    report_lines.append("")
    report_lines.append("Статистики длин документов:")
    report_lines.append(format_mapping({
        "doc_chars_median": float(documents["doc_chars"].median()),
        "doc_chars_p90": float(documents["doc_chars"].quantile(0.90)),
        "doc_chars_max": float(documents["doc_chars"].max()),
        "doc_tokens_median": float(documents["doc_tokens"].median()),
        "doc_tokens_p90": float(documents["doc_tokens"].quantile(0.90)),
        "doc_tokens_max": float(documents["doc_tokens"].max()),
    }))
    report_lines.append("")
    report_lines.append("Дубли и близость документов:")
    report_lines.append(format_mapping({
        "duplicate_text_groups": int(len(duplicate_text_groups)),
        "extra_duplicate_docs": int((duplicate_text_groups["doc_count"] - 1).sum()),
        "median_doc_nn_tfidf_similarity": float(np.median(max_doc_nn_similarity)),
        "share_docs_with_nn_sim_ge_0_8": float((max_doc_nn_similarity >= 0.8).mean()),
        "share_docs_with_nn_sim_ge_0_7": float((max_doc_nn_similarity >= 0.7).mean()),
    }))
    report_lines.append("")
    if not duplicate_text_groups.empty:
        report_lines.append("Крупнейшие группы точных дублей текста:")
        report_lines.append(
            format_table(
                duplicate_text_groups[["doc_count", "doc_ids"]].head(10).reset_index(drop=True),
                index=False,
            )
        )
        report_lines.append("")
    report_lines.append("## 3. Вопросы и разметка train/test")
    report_lines.append("")
    report_lines.append("Статистики длины вопросов:")
    report_lines.append(
        format_mapping(
            {
                "train_q_tokens_median": float(train["q_tokens"].median()),
                "train_q_tokens_p90": float(train["q_tokens"].quantile(0.90)),
                "test_q_tokens_median": float(test["q_tokens"].median()),
                "test_q_tokens_p90": float(test["q_tokens"].quantile(0.90)),
                "train_q_chars_median": float(train["q_chars"].median()),
                "test_q_chars_median": float(test["q_chars"].median()),
            }
        )
    )
    report_lines.append("")
    report_lines.append("Темы train:")
    report_lines.append(format_table(top_topics, index=False))
    report_lines.append("")
    report_lines.append("Покрытие `gold_doc_id` в train:")
    report_lines.append(format_table(gold_coverage, index=False))
    report_lines.append("")
    report_lines.append("Валидность evidence-офсетов:")
    report_lines.append(format_mapping({
        "evidence_offsets_valid": evidence_offset_valid,
        "train_rows": len(train),
        "share_valid": evidence_offset_valid / len(train),
    }))
    report_lines.append("")
    report_lines.append("Статистики по `gold_evidence_text`:")
    report_lines.append(
        format_mapping(
            {
                "evidence_tokens_median": float(train["evidence_tokens"].median()),
                "evidence_tokens_p90": float(train["evidence_tokens"].quantile(0.90)),
                "evidence_span_chars_median": float(train["evidence_span"].median()),
                "evidence_start_ratio_median": float(train["evidence_pos_ratio"].median()),
                "share_evidence_before_25pct": float((train["evidence_pos_ratio"] <= 0.25).mean()),
                "share_evidence_after_75pct": float((train["evidence_pos_ratio"] >= 0.75).mean()),
            }
        )
    )
    report_lines.append("")
    report_lines.append("## 4. Retrieval-сигналы и сложность задачи")
    report_lines.append("")
    report_lines.append("Лексическое совпадение вопроса с правильным документом и с evidence:")
    report_lines.append(
        format_mapping(
            {
                "question_vs_doc_overlap_median": float(train["question_vs_doc_overlap"].median()),
                "question_vs_doc_overlap_p90": float(train["question_vs_doc_overlap"].quantile(0.90)),
                "question_vs_evidence_overlap_median": float(train["question_vs_evidence_overlap"].median()),
                "question_vs_evidence_overlap_p90": float(train["question_vs_evidence_overlap"].quantile(0.90)),
                "share_evidence_overlap_le_0_3": float((train["question_vs_evidence_overlap"] <= 0.30).mean()),
                "share_evidence_overlap_le_0_5": float((train["question_vs_evidence_overlap"] <= 0.50).mean()),
            }
        )
    )
    report_lines.append("")
    report_lines.append("Сходство test-вопросов с train-вопросами:")
    report_lines.append(
        format_mapping(
            {
                "train_vocab_size": float(len(train_vocab)),
                "test_vocab_size": float(len(test_vocab)),
                "test_only_vocab": float(len(test_vocab - train_vocab)),
                "test_unique_token_coverage_by_train_vocab": float(test_token_coverage),
                "median_max_test_to_train_question_tfidf_similarity": float(np.median(max_train_question_similarity)),
                "share_test_questions_with_max_sim_ge_0_5": float((max_train_question_similarity >= 0.5).mean()),
                "share_test_questions_with_max_sim_ge_0_7": float((max_train_question_similarity >= 0.7).mean()),
            }
        )
    )
    report_lines.append("")
    report_lines.append("## 5. Диагностика baseline TF-IDF")
    report_lines.append("")
    report_lines.append(
        format_mapping(
            {
                "baseline_recall_at_5": baseline_recall_at_5,
                "baseline_top1_accuracy": baseline_top1_accuracy,
            }
        )
    )
    report_lines.append("")
    report_lines.append("Recall@5 по типам документов:")
    report_lines.append(format_table(baseline_by_doc_kind.round(3).reset_index(), index=False))
    report_lines.append("")
    report_lines.append("Recall@5 по инстанциям:")
    report_lines.append(format_table(baseline_by_instance.round(3).reset_index(), index=False))
    report_lines.append("")
    report_lines.append("Самые трудные темы для baseline:")
    report_lines.append(
        format_table(
            baseline_by_topic[baseline_by_topic["questions"] >= 8].head(5).round(3).reset_index(),
            index=False,
        )
    )
    report_lines.append("")
    report_lines.append("Самые простые темы для baseline:")
    report_lines.append(
        format_table(
            baseline_by_topic[baseline_by_topic["questions"] >= 8].tail(5).round(3).reset_index(),
            index=False,
        )
    )
    report_lines.append("")
    report_lines.append("Часто встречающиеся, но трудные `gold_doc_id` (минимум 8 вопросов на документ):")
    report_lines.append(format_table(frequent_hard_gold_docs.round(3).reset_index(), index=False))
    report_lines.append("")
    report_lines.append("## 6. Выводы")
    report_lines.append("")
    report_lines.append(
        bullet(
            f"Корпус маленький по размеру ({len(documents)} документов), но очень однотипный: "
            f"обнаружено {int((duplicate_text_groups['doc_count'] - 1).sum())} точных дубликатов сверх первых копий, "
            f"а медианная TF-IDF-близость до ближайшего соседа равна {np.median(max_doc_nn_similarity):.3f}. "
            "Это подтверждает, что основная сложность не в объеме, а в различении почти одинаковых судебных актов."
        )
    )
    report_lines.append(
        bullet(
            f"Train покрывает только {unique_gold_docs} из {len(documents)} документов корпуса; "
            f"{len(documents) - unique_gold_docs} документов ни разу не выступают правильным ответом. "
            "При этом вопросы концентрируются на ограниченном наборе дел, поэтому полезны методы, которые умеют "
            "различать близкие документы внутри одного тематического кластера."
        )
    )
    report_lines.append(
        bullet(
            f"Вопросы короткие (медиана {int(train['q_tokens'].median())} токенов), а evidence-фрагменты длинные "
            f"(медиана {int(train['evidence_tokens'].median())} токенов) и часто лежат далеко от начала документа "
            f"(медианная позиция {train['evidence_pos_ratio'].median():.3f} от длины акта). "
            "Это сильный аргумент в пользу chunking/passage retrieval вместо поиска только по полным документам."
        )
    )
    report_lines.append(
        bullet(
            f"Лексический разрыв существенный: медианный overlap вопроса с `gold_evidence_text` всего "
            f"{train['question_vs_evidence_overlap'].median():.3f}, а в {safe_pct((train['question_vs_evidence_overlap'] <= 0.30).mean())} "
            "случаев совпадает не более 30% токенов вопроса. Простое точное совпадение слов и сырые bag-of-words здесь системно слабы."
        )
    )
    report_lines.append(
        bullet(
            f"TF-IDF baseline ожидаемо ограничен: Recall@5 = {baseline_recall_at_5:.4f}, top-1 accuracy = {baseline_top1_accuracy:.4f}. "
            f"Особенно плохо он работает на темах `персональные данные` ({baseline_by_topic.loc['персональные данные', 'recall_at_5']:.3f}), "
            f"`переход в НПФ / недействительность волеизъявления` ({baseline_by_topic.loc['переход в НПФ / недействительность волеизъявления', 'recall_at_5']:.3f}) "
            f"и `инвестиционный доход и денежные расчеты` ({baseline_by_topic.loc['инвестиционный доход и денежные расчеты', 'recall_at_5']:.3f}). "
            "То есть baseline хуже всего там, где нужен смысловой матчинг или различение очень похожих мотивировок."
        )
    )
    report_lines.append(
        bullet(
            "Следующие улучшения прямо следуют из EDA: лемматизация или BM25 вместо сырого TF-IDF, passage-level retrieval по абзацам, "
            "гибрид sparse+dense, а затем реранжирование top-N кандидатов cross-encoder'ом. "
            "Разметка `gold_evidence_text` подходит для построения именно такого двухэтапного retrieval."
        )
    )
    report_lines.append("")

    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved EDA report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
