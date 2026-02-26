"""Stratified train/val/test splits; persist as .npz."""

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.paths import ensure_dir, get_splits_dir

logger = logging.getLogger(__name__)

TaskType = Literal["multiclass", "multilabel"]


def _stratify_multiclass(y: np.ndarray, test_size: float, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified splits for single-label multiclass y (n_samples,)."""
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    # From rest, split val/test; relative to rest, val fraction = val_size / (1 - (1-test_size)) so that val is val_size of full
    rest_size = len(idx_rest)
    n_val = max(1, int(rest_size * val_size / (1 - (1 - test_size)))) if test_size < 1 else 0
    n_val = min(n_val, rest_size - 1)
    if n_val == 0:
        idx_val = np.array([], dtype=np.int64)
        idx_test = idx_rest
    else:
        y_rest = y[idx_rest]
        idx_val, idx_test = train_test_split(idx_rest, test_size=rest_size - n_val, stratify=y_rest, random_state=seed + 1)
    return idx_train, idx_val, idx_test


def _stratify_multilabel(y: np.ndarray, test_size: float, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pseudo-stratification for multilabel: use first column or sum of labels as stratify target."""
    # Use first label column for stratification; if all same, fall back to second, etc.
    strat = y[:, 0] if y.ndim == 2 else y
    for j in range(y.shape[1] if y.ndim == 2 else 1):
        strat = y[:, j] if y.ndim == 2 else y
        if len(np.unique(strat)) > 1:
            break
    idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(idx, test_size=test_size, stratify=strat, random_state=seed)
    rest_size = len(idx_rest)
    n_val = max(1, int(rest_size * val_size)) if rest_size > 1 else 0
    n_val = min(n_val, rest_size - 1)
    if n_val == 0:
        idx_val = np.array([], dtype=np.int64)
        idx_test = idx_rest
    else:
        strat_rest = strat[idx_rest]
        idx_val, idx_test = train_test_split(idx_rest, test_size=rest_size - n_val, stratify=strat_rest, random_state=seed + 1)
    return idx_train, idx_val, idx_test


def create_splits(
    y: np.ndarray,
    task_type: TaskType = "multiclass",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (train_indices, val_indices, test_indices) as 1D int arrays.
    val_size is fraction of the data used for validation (after taking test out).
    """
    if task_type == "multiclass":
        return _stratify_multiclass(y, test_size, val_size, seed)
    return _stratify_multilabel(y, test_size, val_size, seed)


def save_splits(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    path: Optional[str | Path] = None,
) -> Path:
    """Save split indices to .npz. Default path: data/processed/splits/splits.npz."""
    out_dir = ensure_dir(get_splits_dir())
    if path is None:
        path = out_dir / "splits.npz"
    else:
        path = Path(path)
    np.savez(path, train=idx_train, val=idx_val, test=idx_test)
    logger.info("Saved splits to %s", path)
    return path


def load_splits(path: Optional[str | Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train/val/test indices from .npz."""
    if path is None:
        path = get_splits_dir() / "splits.npz"
    else:
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found: {path}")
    data = np.load(path)
    return data["train"], data["val"], data["test"]
