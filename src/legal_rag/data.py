from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

REQUIRED_COLUMNS = {
    "documents": {"doc_id", "text"},
    "train": {"qid", "question", "gold_doc_id"},
    "test": {"qid", "question"},
    "sample_submission": {"qid", "doc_id"},
}

DATASET_FILES = {
    "documents": "documents.csv",
    "train": "train.csv",
    "test": "test.csv",
    "sample_submission": "sample_submission.csv",
}


@dataclass(frozen=True)
class PathsConfig:
    raw_data_dir: Path
    processed_data_dir: Path
    outputs_dir: Path


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_configured_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return resolve_repo_root() / path


def load_paths_config(config_path: str | Path) -> PathsConfig:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = resolve_repo_root() / config_file

    config = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    missing_keys = {"raw_data_dir", "processed_data_dir", "outputs_dir"} - set(config)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"Missing keys in paths config {config_file}: {missing}")

    return PathsConfig(
        raw_data_dir=_resolve_configured_path(config["raw_data_dir"]),
        processed_data_dir=_resolve_configured_path(config["processed_data_dir"]),
        outputs_dir=_resolve_configured_path(config["outputs_dir"]),
    )


def ensure_output_dirs(paths: PathsConfig) -> None:
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictions", "metrics", "submissions"):
        (paths.outputs_dir / name).mkdir(parents=True, exist_ok=True)


def dataset_path(paths: PathsConfig, dataset_name: str) -> Path:
    if dataset_name not in DATASET_FILES:
        raise KeyError(f"Unknown dataset name: {dataset_name}")
    return paths.raw_data_dir / DATASET_FILES[dataset_name]


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    missing = set(required_columns) - set(dataframe.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"{dataset_name} is missing columns: {missing_columns}")


def _read_dataset(paths: PathsConfig, dataset_name: str) -> pd.DataFrame:
    file_path = dataset_path(paths, dataset_name)
    if not file_path.exists():
        raise FileNotFoundError(f"Required file does not exist: {file_path}")

    dataframe = pd.read_csv(file_path)
    validate_columns(dataframe, REQUIRED_COLUMNS[dataset_name], dataset_name)

    for column in ("qid", "doc_id", "gold_doc_id"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype(str)
    for column in ("question", "text", "ideal_answer", "gold_evidence_text", "topic"):
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].fillna("")

    return dataframe


def load_documents(paths: PathsConfig) -> pd.DataFrame:
    return _read_dataset(paths, "documents")


def load_train(paths: PathsConfig) -> pd.DataFrame:
    return _read_dataset(paths, "train")


def load_test(paths: PathsConfig) -> pd.DataFrame:
    return _read_dataset(paths, "test")


def load_sample_submission(paths: PathsConfig) -> pd.DataFrame:
    return _read_dataset(paths, "sample_submission")
