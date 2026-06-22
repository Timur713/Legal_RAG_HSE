from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from legal_rag.data import (  # noqa: E402
    DATASET_FILES,
    dataset_path,
    load_documents,
    load_paths_config,
    load_sample_submission,
    load_test,
    load_train,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset files and required columns.")
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to YAML config with raw/processed/output directories.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths_config(args.paths)

    print(f"raw_data_dir={paths.raw_data_dir}")
    print(f"processed_data_dir={paths.processed_data_dir}")
    print(f"outputs_dir={paths.outputs_dir}")
    print()

    print("Expected files:")
    for dataset_name in DATASET_FILES:
        path = dataset_path(paths, dataset_name)
        print(f"  - {dataset_name}: {path} ({'ok' if path.exists() else 'missing'})")
    print()

    datasets = {
        "documents": load_documents(paths),
        "train": load_train(paths),
        "test": load_test(paths),
        "sample_submission": load_sample_submission(paths),
    }

    for name, dataframe in datasets.items():
        print(f"{name}: shape={dataframe.shape}")
        print(f"columns={list(dataframe.columns)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
