"""Structured pooling: reduce MultiROCKET features by grouping (kernel_idx, statistic_type, origin)."""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Fallback layout when metadata is not available: origin=2, stat=4, kernels=K -> total = 2*4*K
DEFAULT_N_ORIGINS = 2
DEFAULT_N_STATS = 4


def structured_pooling_fallback(
    X: np.ndarray,
    n_origins: int = DEFAULT_N_ORIGINS,
    n_stats: int = DEFAULT_N_STATS,
    pool: str = "mean",
) -> np.ndarray:
    """
    Deterministic pooling assuming features are ordered [origin, stat, kernel]:
    shape (n_samples, n_origins * n_stats * n_kernels). Reshape to (n_samples, n_origins, n_stats, n_kernels),
    then pool over origins and stats to get (n_samples, n_kernels). Raises if size does not match.
    pool: 'mean' or 'max'.
    """
    n_samples, total = X.shape
    n_kernels = total // (n_origins * n_stats)
    if n_kernels * n_origins * n_stats != total:
        raise ValueError(
            f"Feature dimension {total} does not match n_origins={n_origins} * n_stats={n_stats} * n_kernels. "
            f"Expected total divisible by {n_origins * n_stats}."
        )
    # (n_samples, n_origins, n_stats, n_kernels)
    Y = X.reshape(n_samples, n_origins, n_stats, n_kernels)
    if pool == "mean":
        out = Y.mean(axis=(1, 2))  # (n_samples, n_kernels)
    elif pool == "max":
        out = Y.max(axis=(1, 2))
    else:
        raise ValueError(f"pool must be 'mean' or 'max', got {pool}")
    return np.asarray(out, dtype=np.float32)


def structured_pooling_with_metadata(
    X: np.ndarray,
    group_indices: List[np.ndarray],
    pool: str = "mean",
) -> np.ndarray:
    """
    Pool by explicit groups. group_indices: list of 1D int arrays, each array is indices
    into the feature dim that form one group. Output shape (n_samples, len(group_indices)).
    """
    n_samples = X.shape[0]
    n_groups = len(group_indices)
    out = np.zeros((n_samples, n_groups), dtype=np.float32)
    for g, idx in enumerate(group_indices):
        block = X[:, idx]
        if pool == "mean":
            out[:, g] = block.mean(axis=1)
        elif pool == "max":
            out[:, g] = block.max(axis=1)
        else:
            raise ValueError(f"pool must be 'mean' or 'max', got {pool}")
    return out


def structured_pool(
    X: np.ndarray,
    n_origins: Optional[int] = None,
    n_stats: Optional[int] = None,
    group_indices: Optional[List[np.ndarray]] = None,
    pool: str = "mean",
) -> np.ndarray:
    """
    Structured pooling. If group_indices is provided, use metadata-based pooling.
    Else use fallback with n_origins, n_stats (defaults 2, 4); total features must be n_origins*n_stats*n_kernels.
    """
    if group_indices is not None:
        return structured_pooling_with_metadata(X, group_indices, pool=pool)
    n_origins = n_origins or DEFAULT_N_ORIGINS
    n_stats = n_stats or DEFAULT_N_STATS
    return structured_pooling_fallback(X, n_origins=n_origins, n_stats=n_stats, pool=pool)
