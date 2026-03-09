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


# ---------------------------------------------------------------------------
# Dilation-aware (kernel-informed) pooling
# ---------------------------------------------------------------------------

STAT_NAMES = ["PPV", "LSPV", "MPV", "MIPV"]
ORIGIN_NAMES = ["raw", "differenced"]


def build_dilation_pool_groups(
    dilations: np.ndarray,
    num_features_per_dilation: np.ndarray,
    n_kernels_per_group: int = 84,
    n_stats: int = DEFAULT_N_STATS,
    n_origins: int = DEFAULT_N_ORIGINS,
) -> Tuple[List[np.ndarray], List[dict]]:
    """Build column-index groups for dilation-aware pooling of MultiROCKET features.

    Each group contains all column indices for one (origin, statistic, dilation) triple.
    Within that group there are ``n_kernels_per_group * num_features_per_dilation[d]``
    columns that should be pooled (mean or max) into a single output value.

    MultiROCKET feature layout (for n_base base features, n_stats stats, n_origins origins):
      - Origin 0 (raw):          columns [0, n_base * n_stats)
      - Origin 1 (differenced):  columns [n_base * n_stats, 2 * n_base * n_stats)
      Within each origin block, stat s occupies columns [s * n_base, (s+1) * n_base).
      Within a stat block, base-feature index b runs 0..n_base-1 ordered by
      (dilation_index, kernel_index within dilation).

    Returns:
        group_indices:  list of 1-D int64 arrays (one per output feature)
        group_meta:     parallel list of dicts describing each group
    """
    nfpd = np.asarray(num_features_per_dilation)
    n_base = int(np.sum(n_kernels_per_group * nfpd))

    # base-feature ranges per dilation
    dilation_ranges: List[Tuple[int, int]] = []
    offset = 0
    for d_nfpd in nfpd:
        n = n_kernels_per_group * int(d_nfpd)
        dilation_ranges.append((offset, offset + n))
        offset += n

    n_per_origin = n_stats * n_base  # e.g. 4 * 2016 = 8064

    group_indices: List[np.ndarray] = []
    group_meta: List[dict] = []

    for o_idx in range(n_origins):
        o_off = o_idx * n_per_origin
        for s_idx in range(n_stats):
            s_off = s_idx * n_base
            for d_idx, (d_start, d_end) in enumerate(dilation_ranges):
                cols = np.arange(o_off + s_off + d_start,
                                 o_off + s_off + d_end, dtype=np.int64)
                group_indices.append(cols)
                group_meta.append({
                    "dilation": int(dilations[d_idx]),
                    "dilation_idx": d_idx,
                    "stat_idx": s_idx,
                    "stat_name": STAT_NAMES[s_idx] if s_idx < len(STAT_NAMES) else f"stat{s_idx}",
                    "origin_idx": o_idx,
                    "origin_name": ORIGIN_NAMES[o_idx] if o_idx < len(ORIGIN_NAMES) else f"origin{o_idx}",
                    "group_size": int(d_end - d_start),
                })

    logger.info(
        "Built %d dilation-aware pool groups from %d dilations × %d stats × %d origins "
        "(input_dim=%d → output_dim=%d)",
        len(group_indices), len(dilations), n_stats, n_origins,
        n_origins * n_per_origin, len(group_indices),
    )
    return group_indices, group_meta
