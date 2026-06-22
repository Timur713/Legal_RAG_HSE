"""Core helpers for the Legal RAG hackathon scaffold."""

from .baseline import TfidfRetriever, build_ranked_predictions
from .data import (
    PathsConfig,
    ensure_output_dirs,
    load_documents,
    load_paths_config,
    load_sample_submission,
    load_test,
    load_train,
)
from .evaluation import recall_at_k
from .submission import create_submission, validate_submission

__all__ = [
    "PathsConfig",
    "TfidfRetriever",
    "build_ranked_predictions",
    "create_submission",
    "ensure_output_dirs",
    "load_documents",
    "load_paths_config",
    "load_sample_submission",
    "load_test",
    "load_train",
    "recall_at_k",
    "validate_submission",
]
