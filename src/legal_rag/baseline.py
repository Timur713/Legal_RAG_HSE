from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .preprocessing import normalize_text, tokenize


@dataclass
class TfidfRetriever:
    top_k: int = 5
    vectorizer: TfidfVectorizer = field(init=False)
    doc_ids: list[str] = field(init=False, default_factory=list)
    doc_matrix: object = field(init=False, default=None)

    def fit(self, documents: pd.DataFrame) -> "TfidfRetriever":
        required_columns = {"doc_id", "text"}
        missing = required_columns - set(documents.columns)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(f"Documents dataframe is missing columns: {missing_columns}")

        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            lowercase=False,
            token_pattern=None,
        )
        self.doc_ids = documents["doc_id"].astype(str).tolist()
        normalized_documents = documents["text"].map(normalize_text)
        self.doc_matrix = self.vectorizer.fit_transform(normalized_documents)
        return self

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.doc_matrix is None:
            raise ValueError("Retriever is not fitted")

        limit = min(top_k or self.top_k, len(self.doc_ids))
        normalized_questions = [normalize_text(question) for question in questions]
        question_matrix = self.vectorizer.transform(normalized_questions)
        similarities = linear_kernel(question_matrix, self.doc_matrix)
        ranked_indices = np.argsort(-similarities, axis=1)[:, :limit]
        ranked_scores = np.take_along_axis(similarities, ranked_indices, axis=1)

        ranked_doc_ids = [[self.doc_ids[index] for index in row] for row in ranked_indices]
        score_rows = [[float(score) for score in row] for row in ranked_scores]
        return ranked_doc_ids, score_rows


def build_ranked_predictions(
    qids: Sequence[object],
    ranked_doc_ids: Sequence[Sequence[object]],
    ranked_scores: Sequence[Sequence[float]] | None = None,
) -> pd.DataFrame:
    if len(qids) != len(ranked_doc_ids):
        raise ValueError("qids and ranked_doc_ids must have the same length")
    if ranked_scores is not None and len(ranked_scores) != len(ranked_doc_ids):
        raise ValueError("ranked_scores and ranked_doc_ids must have the same length")

    rows: list[dict[str, object]] = []
    for index, qid in enumerate(qids):
        doc_ids = ranked_doc_ids[index]
        scores = ranked_scores[index] if ranked_scores is not None else [None] * len(doc_ids)
        for rank, (doc_id, score) in enumerate(zip(doc_ids, scores), start=1):
            row = {"qid": str(qid), "rank": rank, "doc_id": str(doc_id)}
            if score is not None:
                row["score"] = float(score)
            rows.append(row)

    return pd.DataFrame(rows)
