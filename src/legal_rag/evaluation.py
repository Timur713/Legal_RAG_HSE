from __future__ import annotations

from typing import Iterable, Sequence


def deduplicate_top_k(doc_ids: Iterable[object], k: int = 5) -> list[str]:
    unique_doc_ids: list[str] = []
    seen: set[str] = set()

    for raw_doc_id in doc_ids:
        doc_id = str(raw_doc_id)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique_doc_ids.append(doc_id)
        if len(unique_doc_ids) == k:
            break

    return unique_doc_ids


def recall_at_k(
    gold_doc_ids: Sequence[object],
    predictions: Sequence[Sequence[object]],
    k: int = 5,
) -> float:
    if len(gold_doc_ids) != len(predictions):
        raise ValueError("gold_doc_ids and predictions must have the same length")
    if not gold_doc_ids:
        return 0.0

    hits = 0
    for gold_doc_id, predicted_doc_ids in zip(gold_doc_ids, predictions):
        if str(gold_doc_id) in deduplicate_top_k(predicted_doc_ids, k=k):
            hits += 1

    return hits / len(gold_doc_ids)
