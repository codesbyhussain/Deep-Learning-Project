"""Data I/O. Implement load_raw_dataset() here — this is the single place for dataset loading."""

import logging
from typing import Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_raw_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the raw ECG dataset (time series and labels).

    YOU MUST IMPLEMENT THIS FUNCTION in this file. Do not generate dummy data.

    Expected return:
        X : np.ndarray of shape (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps).
            Float time series. If 3D, channels are typically treated as separate series or
            concatenated per sample depending on the feature extractor.
        y : np.ndarray of shape (n_samples,) for single-label multiclass (int labels 0..n_classes-1),
            or (n_samples, n_classes) for multilabel (binary matrix).

    Data location:
        Place raw data files under: data/raw/
        (See README and project_root from src.utils.paths.)

    Example (pseudocode):
        from pathlib import Path
        from src.utils.paths import get_raw_data_dir
        raw_dir = get_raw_data_dir()
        # Load your CSV/parquet/npy etc. from raw_dir
        # X = ...  # shape (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps)
        # y = ...  # shape (n_samples,) or (n_samples, n_classes)
        return X, y
    """
    raise NotImplementedError(
        "load_raw_dataset() is not implemented. Implement it in src/data/io.py: "
        "load your ECG time series and labels from data/raw/ and return (X, y) as described in the docstring."
    )


def save_labels(y: np.ndarray, path: str | None = None) -> None:
    """Save label array to data/processed/labels/ as .npy. Filename from path or default."""
    from pathlib import Path

    from src.utils.paths import get_labels_dir, ensure_dir

    out_dir = ensure_dir(get_labels_dir())
    if path is None:
        path = out_dir / "labels.npy"
    else:
        path = Path(path)
    np.save(path, y)
    logger.info("Saved labels to %s", path)


def load_labels(path: str | None = None) -> np.ndarray:
    """Load labels from .npy. Default path: data/processed/labels/labels.npy."""
    from pathlib import Path

    from src.utils.paths import get_labels_dir

    if path is None:
        path = get_labels_dir() / "labels.npy"
    else:
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    return np.load(path)
