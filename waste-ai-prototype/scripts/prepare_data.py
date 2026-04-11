from __future__ import annotations

from pathlib import Path
import shutil

from src.config import DEFAULT_SOURCE_DATASET, RAW_DATASET_PATH
from src.utils import ensure_directories


def prepare_raw_dataset(source_path: str | Path | None = None) -> None:
    ensure_directories()
    source = Path(source_path) if source_path else DEFAULT_SOURCE_DATASET

    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset not found at {source}. Update the path in src/config.py if needed."
        )

    shutil.copyfile(source, RAW_DATASET_PATH)
    print(f"Copied dataset to: {RAW_DATASET_PATH}")


if __name__ == "__main__":
    prepare_raw_dataset()
