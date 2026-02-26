"""Project root and path helpers for data, models, and experiment outputs."""

import os
from pathlib import Path
from typing import Optional

# Project root: directory containing this file is src/utils, so project_root is parent of src
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parent.parent
PROJECT_ROOT: Path = _SRC_DIR.parent


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Return data/ directory."""
    return PROJECT_ROOT / "data"


def get_raw_data_dir() -> Path:
    """Return data/raw/."""
    return get_data_dir() / "raw"


def get_interim_data_dir() -> Path:
    """Return data/interim/."""
    return get_data_dir() / "interim"


def get_processed_data_dir() -> Path:
    """Return data/processed/."""
    return get_data_dir() / "processed"


def get_splits_dir() -> Path:
    """Return data/processed/splits/."""
    return get_processed_data_dir() / "splits"


def get_multirocket_features_dir() -> Path:
    """Return data/processed/multirocket/."""
    return get_processed_data_dir() / "multirocket"


def get_reduced_dir() -> Path:
    """Return data/processed/reduced/."""
    return get_processed_data_dir() / "reduced"


def get_labels_dir() -> Path:
    """Return data/processed/labels/."""
    return get_processed_data_dir() / "labels"


def get_models_dir() -> Path:
    """Return models/ at project root."""
    return PROJECT_ROOT / "models"


def get_experiment_dir(condition: str) -> Path:
    """Return experiments/<condition>/ (e.g. A1_autoencoder_mlp)."""
    return PROJECT_ROOT / "experiments" / condition


def get_experiment_output_dir(
    condition: str,
    *,
    logs: bool = False,
    checkpoints: bool = False,
) -> Path:
    """Return experiment subdir; create if needed. Use logs=True or checkpoints=True for subdirs."""
    base = get_experiment_dir(condition)
    if logs:
        return base / "logs"
    if checkpoints:
        return base / "checkpoints"
    return base


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
