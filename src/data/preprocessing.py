"""Preprocessing utilities for ECG/time series (e.g. optional normalization)."""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def ensure_3d(X: np.ndarray) -> np.ndarray:
    """
    Ensure time series is 3D: (n_samples, n_channels, n_timesteps).
    If 2D (n_samples, n_timesteps), add channel dim -> (n_samples, 1, n_timesteps).
    """
    if X.ndim == 2:
        return X[:, np.newaxis, :]
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def normalize_per_sample(X: np.ndarray, axis: int = -1) -> np.ndarray:
    """Z-score normalize per sample along axis (default: last = time). Returns float64 copy."""
    out = np.asarray(X, dtype=np.float64)
    mean = out.mean(axis=axis, keepdims=True)
    std = out.std(axis=axis, keepdims=True)
    np.putmask(std, std == 0, 1.0)
    out = (out - mean) / std
    return out
