from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .baseline import TfidfRetriever, build_ranked_predictions
from .preprocessing import normalize_text, tokenize_with_options

DOCUMENT_COLUMNS = {"doc_id", "text"}


@runtime_checkable
class RetrieverProtocol(Protocol):
    top_k: int

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "RetrieverProtocol":
        ...

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        ...


def validate_documents(documents: pd.DataFrame) -> None:
    missing = DOCUMENT_COLUMNS - set(documents.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Documents dataframe is missing columns: {missing_columns}")


def resolve_chunk_stride(chunk_size: int, chunk_stride: int | None) -> int:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    stride = chunk_stride if chunk_stride is not None else chunk_size // 2
    if stride <= 0:
        raise ValueError("chunk_stride must be positive")
    if stride > chunk_size:
        raise ValueError("chunk_stride cannot be greater than chunk_size")
    return stride


def iter_character_chunks(
    text: object,
    *,
    chunk_size: int,
    chunk_stride: int,
) -> list[str]:
    raw_text = str(text)
    if not raw_text:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(raw_text)
    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk = raw_text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_stride
    return chunks


@dataclass
class DocumentChunkStore:
    chunk_size: int = 1600
    chunk_stride: int = 800
    use_lemmas: bool = False
    doc_ids: list[str] = field(init=False, default_factory=list)
    doc_id_to_index: dict[str, int] = field(init=False, default_factory=dict)
    chunks_by_doc_index: list[list[str]] = field(init=False, default_factory=list)
    chunk_bm25_by_doc_index: list[BM25Okapi | None] = field(init=False, default_factory=list)

    def fit(self, documents: pd.DataFrame) -> "DocumentChunkStore":
        validate_documents(documents)
        self.chunk_stride = resolve_chunk_stride(self.chunk_size, self.chunk_stride)
        self.doc_ids = documents["doc_id"].astype(str).tolist()
        self.doc_id_to_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        self.chunks_by_doc_index = []
        self.chunk_bm25_by_doc_index = []

        for text in documents["text"].tolist():
            chunks = iter_character_chunks(
                text,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride,
            )
            if not chunks:
                chunks = [str(text)]
            tokenized_chunks = [
                tokenize_with_options(chunk, use_lemmas=self.use_lemmas)
                for chunk in chunks
            ]
            if any(tokenized_chunks):
                bm25 = BM25Okapi([
                    tokens if tokens else ["__empty__"]
                    for tokens in tokenized_chunks
                ])
            else:
                bm25 = None
            self.chunks_by_doc_index.append(chunks)
            self.chunk_bm25_by_doc_index.append(bm25)
        return self

    def top_chunks(
        self,
        doc_id: str,
        question: object,
        *,
        limit: int = 1,
    ) -> list[str]:
        doc_index = self.doc_id_to_index[str(doc_id)]
        chunks = self.chunks_by_doc_index[doc_index]
        bm25 = self.chunk_bm25_by_doc_index[doc_index]
        if limit <= 0:
            raise ValueError("limit must be positive.")
        if len(chunks) <= limit or bm25 is None:
            return chunks[:limit]

        query_tokens = tokenize_with_options(question, use_lemmas=self.use_lemmas)
        if not query_tokens:
            return chunks[:limit]

        scores = bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(-np.asarray(scores, dtype=float))[:limit]
        return [chunks[int(index)] for index in ranked_indices]

    def best_chunk(self, doc_id: str, question: object) -> str:
        return self.top_chunks(doc_id, question, limit=1)[0]


@dataclass
class ChunkedTfidfRetriever:
    top_k: int = 5
    chunk_size: int = 1600
    chunk_stride: int = 800
    use_lemmas: bool = False
    vectorizer: TfidfVectorizer = field(init=False)
    doc_ids: list[str] = field(init=False, default_factory=list)
    chunk_matrix: object = field(init=False, default=None)
    chunk_doc_indices: np.ndarray = field(init=False, default_factory=lambda: np.array([], dtype=np.int32))
    num_chunks: int = field(init=False, default=0)

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "ChunkedTfidfRetriever":
        del train_queries

        validate_documents(documents)
        self.chunk_stride = resolve_chunk_stride(self.chunk_size, self.chunk_stride)
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda text: tokenize_with_options(text, use_lemmas=self.use_lemmas),
            lowercase=False,
            token_pattern=None,
        )
        self.doc_ids = documents["doc_id"].astype(str).tolist()

        chunk_texts: list[str] = []
        chunk_doc_indices: list[int] = []
        for doc_index, text in enumerate(documents["text"].tolist()):
            for chunk in iter_character_chunks(
                text,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride,
            ):
                chunk_texts.append(normalize_text(chunk))
                chunk_doc_indices.append(doc_index)

        if not chunk_texts:
            raise ValueError("No non-empty chunks were generated from the documents.")

        self.chunk_matrix = self.vectorizer.fit_transform(chunk_texts)
        self.chunk_doc_indices = np.asarray(chunk_doc_indices, dtype=np.int32)
        self.num_chunks = len(chunk_texts)
        return self

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.chunk_matrix is None:
            raise ValueError("Retriever is not fitted")

        limit = min(top_k or self.top_k, len(self.doc_ids))
        normalized_questions = [normalize_text(question) for question in questions]
        question_matrix = self.vectorizer.transform(normalized_questions)
        chunk_similarities = linear_kernel(question_matrix, self.chunk_matrix)

        ranked_doc_ids: list[list[str]] = []
        ranked_scores: list[list[float]] = []
        num_docs = len(self.doc_ids)
        for row in chunk_similarities:
            doc_scores = np.full(num_docs, -np.inf, dtype=float)
            np.maximum.at(doc_scores, self.chunk_doc_indices, np.asarray(row, dtype=float))
            ranked_indices = np.argsort(-doc_scores)[:limit]
            ranked_doc_ids.append([self.doc_ids[index] for index in ranked_indices])
            ranked_scores.append([float(doc_scores[index]) for index in ranked_indices])

        return ranked_doc_ids, ranked_scores


@dataclass
class Bm25Retriever:
    top_k: int = 5
    use_lemmas: bool = False
    doc_ids: list[str] = field(init=False, default_factory=list)
    bm25: BM25Okapi | None = field(init=False, default=None)

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "Bm25Retriever":
        del train_queries

        validate_documents(documents)
        self.doc_ids = documents["doc_id"].astype(str).tolist()
        tokenized_documents = [
            tokenize_with_options(text, use_lemmas=self.use_lemmas)
            for text in documents["text"].tolist()
        ]
        if not any(tokenized_documents):
            raise ValueError("No non-empty tokenized documents were generated.")

        self.bm25 = BM25Okapi(tokenized_documents)
        return self

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.bm25 is None:
            raise ValueError("Retriever is not fitted")

        limit = min(top_k or self.top_k, len(self.doc_ids))
        ranked_doc_ids: list[list[str]] = []
        ranked_scores: list[list[float]] = []

        for question in questions:
            query_tokens = tokenize_with_options(question, use_lemmas=self.use_lemmas)
            doc_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
            ranked_indices = np.argsort(-doc_scores)[:limit]
            ranked_doc_ids.append([self.doc_ids[index] for index in ranked_indices])
            ranked_scores.append([float(doc_scores[index]) for index in ranked_indices])

        return ranked_doc_ids, ranked_scores


@dataclass
class ChunkedBm25Retriever:
    top_k: int = 5
    chunk_size: int = 1600
    chunk_stride: int = 800
    use_lemmas: bool = False
    doc_ids: list[str] = field(init=False, default_factory=list)
    bm25: BM25Okapi | None = field(init=False, default=None)
    chunk_doc_indices: np.ndarray = field(init=False, default_factory=lambda: np.array([], dtype=np.int32))
    num_chunks: int = field(init=False, default=0)

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "ChunkedBm25Retriever":
        del train_queries

        validate_documents(documents)
        self.chunk_stride = resolve_chunk_stride(self.chunk_size, self.chunk_stride)
        self.doc_ids = documents["doc_id"].astype(str).tolist()

        tokenized_chunks: list[list[str]] = []
        chunk_doc_indices: list[int] = []
        for doc_index, text in enumerate(documents["text"].tolist()):
            for chunk in iter_character_chunks(
                text,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride,
            ):
                tokens = tokenize_with_options(chunk, use_lemmas=self.use_lemmas)
                if not tokens:
                    continue
                tokenized_chunks.append(tokens)
                chunk_doc_indices.append(doc_index)

        if not tokenized_chunks:
            raise ValueError("No non-empty tokenized chunks were generated from the documents.")

        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunk_doc_indices = np.asarray(chunk_doc_indices, dtype=np.int32)
        self.num_chunks = len(tokenized_chunks)
        return self

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.bm25 is None:
            raise ValueError("Retriever is not fitted")

        limit = min(top_k or self.top_k, len(self.doc_ids))
        ranked_doc_ids: list[list[str]] = []
        ranked_scores: list[list[float]] = []
        num_docs = len(self.doc_ids)

        for question in questions:
            query_tokens = tokenize_with_options(question, use_lemmas=self.use_lemmas)
            chunk_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
            doc_scores = np.full(num_docs, -np.inf, dtype=float)
            np.maximum.at(doc_scores, self.chunk_doc_indices, chunk_scores)
            ranked_indices = np.argsort(-doc_scores)[:limit]
            ranked_doc_ids.append([self.doc_ids[index] for index in ranked_indices])
            ranked_scores.append([float(doc_scores[index]) for index in ranked_indices])

        return ranked_doc_ids, ranked_scores


@dataclass
class ReciprocalRankFusionRetriever:
    component_names: tuple[str, ...]
    top_k: int = 5
    chunk_size: int = 1600
    chunk_stride: int = 800
    use_lemmas: bool = False
    rrf_k: int = 60
    retrievers: list[RetrieverProtocol] = field(init=False, default_factory=list)

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "ReciprocalRankFusionRetriever":
        if not self.component_names:
            raise ValueError("hybrid_rrf requires at least one component retriever.")

        self.retrievers = []
        for name in self.component_names:
            if name == "hybrid_rrf":
                raise ValueError("hybrid_rrf cannot include itself as a component retriever.")
            retriever = create_retriever(
                name,
                top_k=self.top_k,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride,
                use_lemmas=self.use_lemmas,
            )
            retriever.fit(documents, train_queries=train_queries)
            self.retrievers.append(retriever)
        return self

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if not self.retrievers:
            raise ValueError("Retriever is not fitted")

        limit = top_k or self.top_k
        component_rankings = [retriever.retrieve(questions, top_k=limit)[0] for retriever in self.retrievers]

        fused_doc_ids: list[list[str]] = []
        fused_scores: list[list[float]] = []
        for query_index in range(len(questions)):
            scores: dict[str, float] = {}
            for ranking in component_rankings:
                for rank, doc_id in enumerate(ranking[query_index], start=1):
                    scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
            fused_doc_ids.append([doc_id for doc_id, _ in ranked])
            fused_scores.append([float(score) for _, score in ranked])

        return fused_doc_ids, fused_scores


@dataclass
class CrossEncoderRerankerRetriever:
    first_stage_name: str
    top_k: int = 5
    rerank_top_k: int = 20
    chunk_size: int = 1600
    chunk_stride: int = 800
    use_lemmas: bool = False
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_device: str | None = None
    reranker_chunks_per_doc: int = 1
    reranker_combine_mode: str = "ce_only"
    reranker_ce_weight: float = 0.7
    reranker_sparse_weight: float = 0.3
    reranker_rrf_k: int = 60
    first_stage_hybrid_retrievers: tuple[str, ...] = ()
    first_stage_rrf_k: int = 60
    first_stage_retriever: RetrieverProtocol = field(init=False)
    chunk_store: DocumentChunkStore = field(init=False)
    cross_encoder: Any = field(init=False, default=None)

    def fit(
        self,
        documents: pd.DataFrame,
        train_queries: pd.DataFrame | None = None,
    ) -> "CrossEncoderRerankerRetriever":
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive.")
        if self.reranker_batch_size <= 0:
            raise ValueError("reranker_batch_size must be positive.")
        if self.reranker_chunks_per_doc <= 0:
            raise ValueError("reranker_chunks_per_doc must be positive.")
        if self.reranker_combine_mode not in {"ce_only", "rrf", "linear"}:
            raise ValueError("reranker_combine_mode must be one of: ce_only, rrf, linear.")
        if self.reranker_ce_weight < 0 or self.reranker_sparse_weight < 0:
            raise ValueError("reranker_ce_weight and reranker_sparse_weight must be non-negative.")
        if self.reranker_ce_weight == 0 and self.reranker_sparse_weight == 0:
            raise ValueError("At least one reranker weight must be positive.")
        if self.reranker_rrf_k <= 0:
            raise ValueError("reranker_rrf_k must be positive.")

        self.first_stage_retriever = create_retriever(
            self.first_stage_name,
            top_k=self.rerank_top_k,
            chunk_size=self.chunk_size,
            chunk_stride=self.chunk_stride,
            use_lemmas=self.use_lemmas,
            hybrid_retrievers=self.first_stage_hybrid_retrievers,
            rrf_k=self.first_stage_rrf_k,
        )
        self.first_stage_retriever.fit(documents, train_queries=train_queries)

        self.chunk_store = DocumentChunkStore(
            chunk_size=self.chunk_size,
            chunk_stride=self.chunk_stride,
            use_lemmas=self.use_lemmas,
        ).fit(documents)

        from sentence_transformers import CrossEncoder

        encoder_kwargs: dict[str, Any] = {
            "model_name_or_path": self.reranker_model_name,
            "max_length": self.reranker_max_length,
        }
        if self.reranker_device is not None:
            encoder_kwargs["device"] = self.reranker_device
        self.cross_encoder = CrossEncoder(**encoder_kwargs)
        return self

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        minimum = float(np.min(scores))
        maximum = float(np.max(scores))
        if maximum <= minimum:
            return np.ones_like(scores, dtype=float)
        return (scores - minimum) / (maximum - minimum)

    def _combine_scores(
        self,
        doc_ids: list[str],
        ce_scores: np.ndarray,
        sparse_scores: np.ndarray,
    ) -> np.ndarray:
        if self.reranker_combine_mode == "ce_only":
            return ce_scores

        if self.reranker_combine_mode == "linear":
            normalized_ce = self._minmax_normalize(ce_scores)
            normalized_sparse = self._minmax_normalize(sparse_scores)
            return (
                self.reranker_ce_weight * normalized_ce
                + self.reranker_sparse_weight * normalized_sparse
            )

        combined = np.zeros(len(doc_ids), dtype=float)
        ce_ranking = sorted(
            range(len(doc_ids)),
            key=lambda index: (-float(ce_scores[index]), doc_ids[index]),
        )
        sparse_ranking = sorted(
            range(len(doc_ids)),
            key=lambda index: (-float(sparse_scores[index]), doc_ids[index]),
        )
        for rank, index in enumerate(ce_ranking, start=1):
            combined[index] += self.reranker_ce_weight / (self.reranker_rrf_k + rank)
        for rank, index in enumerate(sparse_ranking, start=1):
            combined[index] += self.reranker_sparse_weight / (self.reranker_rrf_k + rank)
        return combined

    def retrieve(
        self,
        questions: Sequence[object],
        top_k: int | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.cross_encoder is None:
            raise ValueError("Retriever is not fitted")

        limit = min(top_k or self.top_k, self.rerank_top_k)
        first_stage_doc_ids, first_stage_scores = self.first_stage_retriever.retrieve(
            questions,
            top_k=self.rerank_top_k,
        )

        pairs: list[tuple[str, str]] = []
        candidate_counts: list[int] = []
        chunk_counts_per_doc: list[list[int]] = []
        for question, doc_ids in zip(questions, first_stage_doc_ids):
            candidate_counts.append(len(doc_ids))
            query_chunk_counts: list[int] = []
            for doc_id in doc_ids:
                top_chunks = self.chunk_store.top_chunks(
                    str(doc_id),
                    question,
                    limit=self.reranker_chunks_per_doc,
                )
                query_chunk_counts.append(len(top_chunks))
                for chunk_text in top_chunks:
                    pairs.append((str(question), chunk_text))
            chunk_counts_per_doc.append(query_chunk_counts)

        if not pairs:
            return [[] for _ in questions], [[] for _ in questions]

        raw_scores = self.cross_encoder.predict(
            pairs,
            batch_size=self.reranker_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)

        ranked_doc_ids: list[list[str]] = []
        ranked_scores: list[list[float]] = []
        offset = 0
        for doc_ids, sparse_scores, chunk_counts in zip(
            first_stage_doc_ids,
            first_stage_scores,
            chunk_counts_per_doc,
            strict=True,
        ):
            ce_doc_scores: list[float] = []
            for chunk_count in chunk_counts:
                doc_chunk_scores = scores[offset : offset + chunk_count]
                offset += chunk_count
                ce_doc_scores.append(float(np.max(doc_chunk_scores)))
            final_scores = self._combine_scores(
                list(doc_ids),
                np.asarray(ce_doc_scores, dtype=float),
                np.asarray(sparse_scores, dtype=float),
            )
            ranked = sorted(
                zip(doc_ids, final_scores, strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )[:limit]
            ranked_doc_ids.append([str(doc_id) for doc_id, _ in ranked])
            ranked_scores.append([float(score) for _, score in ranked])

        return ranked_doc_ids, ranked_scores


AVAILABLE_RETRIEVERS = (
    "tfidf",
    "bm25",
    "chunked_tfidf",
    "chunked_bm25",
    "hybrid_rrf",
    "cross_encoder_rerank",
)


def create_retriever(
    name: str,
    *,
    top_k: int,
    chunk_size: int = 1600,
    chunk_stride: int | None = None,
    use_lemmas: bool = False,
    hybrid_retrievers: Sequence[str] | None = None,
    rrf_k: int = 60,
    rerank_top_k: int = 20,
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 512,
    reranker_device: str | None = None,
    reranker_chunks_per_doc: int = 1,
    reranker_combine_mode: str = "ce_only",
    reranker_ce_weight: float = 0.7,
    reranker_sparse_weight: float = 0.3,
    reranker_rrf_k: int = 60,
    first_stage_retriever: str = "hybrid_rrf",
    first_stage_hybrid_retrievers: Sequence[str] | None = None,
    first_stage_rrf_k: int = 60,
) -> RetrieverProtocol:
    if name == "tfidf":
        return TfidfRetriever(top_k=top_k, use_lemmas=use_lemmas)
    if name == "bm25":
        return Bm25Retriever(top_k=top_k, use_lemmas=use_lemmas)
    if name == "chunked_tfidf":
        return ChunkedTfidfRetriever(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_stride=resolve_chunk_stride(chunk_size, chunk_stride),
            use_lemmas=use_lemmas,
        )
    if name == "chunked_bm25":
        return ChunkedBm25Retriever(
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_stride=resolve_chunk_stride(chunk_size, chunk_stride),
            use_lemmas=use_lemmas,
        )
    if name == "hybrid_rrf":
        component_names = tuple(str(component) for component in (hybrid_retrievers or ()))
        return ReciprocalRankFusionRetriever(
            component_names=component_names,
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_stride=resolve_chunk_stride(chunk_size, chunk_stride),
            use_lemmas=use_lemmas,
            rrf_k=rrf_k,
        )
    if name == "cross_encoder_rerank":
        return CrossEncoderRerankerRetriever(
            first_stage_name=first_stage_retriever,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            chunk_size=chunk_size,
            chunk_stride=resolve_chunk_stride(chunk_size, chunk_stride),
            use_lemmas=use_lemmas,
            reranker_model_name=reranker_model_name,
            reranker_batch_size=reranker_batch_size,
            reranker_max_length=reranker_max_length,
            reranker_device=reranker_device,
            reranker_chunks_per_doc=reranker_chunks_per_doc,
            reranker_combine_mode=reranker_combine_mode,
            reranker_ce_weight=reranker_ce_weight,
            reranker_sparse_weight=reranker_sparse_weight,
            reranker_rrf_k=reranker_rrf_k,
            first_stage_hybrid_retrievers=tuple(str(component) for component in (first_stage_hybrid_retrievers or ())),
            first_stage_rrf_k=first_stage_rrf_k,
        )
    raise ValueError(f"Unknown retriever: {name}")


def get_retriever_params(
    name: str,
    *,
    chunk_size: int = 1600,
    chunk_stride: int | None = None,
    use_lemmas: bool = False,
    hybrid_retrievers: Sequence[str] | None = None,
    rrf_k: int = 60,
    rerank_top_k: int = 20,
    reranker_model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 512,
    reranker_device: str | None = None,
    reranker_chunks_per_doc: int = 1,
    reranker_combine_mode: str = "ce_only",
    reranker_ce_weight: float = 0.7,
    reranker_sparse_weight: float = 0.3,
    reranker_rrf_k: int = 60,
    first_stage_retriever: str = "hybrid_rrf",
    first_stage_hybrid_retrievers: Sequence[str] | None = None,
    first_stage_rrf_k: int = 60,
) -> dict[str, Any]:
    if name in {"tfidf", "bm25"}:
        return {"name": name, "use_lemmas": bool(use_lemmas)}
    if name in {"chunked_tfidf", "chunked_bm25"}:
        return {
            "name": name,
            "chunk_size": int(chunk_size),
            "chunk_stride": int(resolve_chunk_stride(chunk_size, chunk_stride)),
            "use_lemmas": bool(use_lemmas),
        }
    if name == "hybrid_rrf":
        return {
            "name": name,
            "chunk_size": int(chunk_size),
            "chunk_stride": int(resolve_chunk_stride(chunk_size, chunk_stride)),
            "use_lemmas": bool(use_lemmas),
            "hybrid_retrievers": [str(component) for component in (hybrid_retrievers or ())],
            "rrf_k": int(rrf_k),
        }
    if name == "cross_encoder_rerank":
        return {
            "name": name,
            "chunk_size": int(chunk_size),
            "chunk_stride": int(resolve_chunk_stride(chunk_size, chunk_stride)),
            "use_lemmas": bool(use_lemmas),
            "rerank_top_k": int(rerank_top_k),
            "reranker_model_name": str(reranker_model_name),
            "reranker_batch_size": int(reranker_batch_size),
            "reranker_max_length": int(reranker_max_length),
            "reranker_device": reranker_device,
            "reranker_chunks_per_doc": int(reranker_chunks_per_doc),
            "reranker_combine_mode": str(reranker_combine_mode),
            "reranker_ce_weight": float(reranker_ce_weight),
            "reranker_sparse_weight": float(reranker_sparse_weight),
            "reranker_rrf_k": int(reranker_rrf_k),
            "first_stage_retriever": str(first_stage_retriever),
            "first_stage_hybrid_retrievers": [
                str(component) for component in (first_stage_hybrid_retrievers or ())
            ],
            "first_stage_rrf_k": int(first_stage_rrf_k),
        }
    raise ValueError(f"Unknown retriever: {name}")


__all__ = [
    "AVAILABLE_RETRIEVERS",
    "Bm25Retriever",
    "ChunkedBm25Retriever",
    "ChunkedTfidfRetriever",
    "CrossEncoderRerankerRetriever",
    "DocumentChunkStore",
    "ReciprocalRankFusionRetriever",
    "RetrieverProtocol",
    "TfidfRetriever",
    "build_ranked_predictions",
    "create_retriever",
    "get_retriever_params",
    "iter_character_chunks",
    "resolve_chunk_stride",
    "validate_documents",
]
