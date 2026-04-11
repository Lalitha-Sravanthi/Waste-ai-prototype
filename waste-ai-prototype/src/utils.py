from __future__ import annotations

from pathlib import Path

from src.config import ASSETS_DIR, MODELS_DIR, PROCESSED_DIR, RAW_DIR


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def file_exists(path: str | Path) -> bool:
    return Path(path).exists()
