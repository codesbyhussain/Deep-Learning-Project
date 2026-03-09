"""MultiROCKET wrapper: use existing package (sktime preferred); fit on train, batch transform, persist."""

import logging
import time
from pathlib import Path
from typing import Any, Optional, Protocol, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class MultiRocketAdapter(Protocol):
    """Adapter interface so a different MultiROCKET implementation can be plugged in."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "MultiRocketAdapter":
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


def _create_sktime_multirocket(num_kernels: int = 2048, seed: int = 0) -> Any:
    """Create MultiRocketMultivariate transformer from sktime. Requires sktime with MultiRocketMultivariate (no univariate fallback)."""
    try:
        from sktime.transformations.panel.rocket import MultiRocketMultivariate

        return MultiRocketMultivariate(num_kernels=num_kernels, random_state=seed)
    except ImportError as e:
        raise ImportError(
            "MultiROCKET multivariate requires sktime with MultiRocketMultivariate. "
            "Install with: pip install sktime. Or pass a custom adapter to create_multirocket_transformer(adapter=...)."
        ) from e


def create_multirocket_transformer(
    num_kernels: int = 2048,
    seed: int = 0,
    adapter: Optional[MultiRocketAdapter] = None,
) -> Any:
    """
    Create MultiROCKET transformer. Prefer sktime MultiRocket; else use provided adapter.
    adapter: optional external implementation to use instead of sktime.
    """
    if adapter is not None:
        return adapter
    trans = _create_sktime_multirocket(num_kernels=num_kernels, seed=seed)
    return trans


def fit_multirocket(
    transformer: Any,
    X_train: np.ndarray,
) -> Any:
    """Fit MultiROCKET on training time series only. X_train: (n_samples, n_timesteps) or (n_samples, n_channels, n_timesteps)."""
    # sktime expects 3D (n_samples, n_channels, n_timesteps) for panel
    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]
    transformer.fit(X_train)
    logger.info("Fitted MultiROCKET on train shape %s", X_train.shape)
    return transformer


def transform_multirocket_batched(
    transformer: Any,
    X: np.ndarray,
    batch_size: int = 1024,
) -> np.ndarray:
    """Transform in batches; return (n_samples, n_features) float array. Logs progress per batch."""
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    # sktime's numba-backed MultiRocketMultivariate expects float64 input (fit uses astype(float64))
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    n_batches = (n + batch_size - 1) // batch_size
    chunks = []
    t0 = time.time()
    for i, start in enumerate(range(0, n, batch_size)):
        end = min(start + batch_size, n)
        chunk = transformer.transform(X[start:end])
        if hasattr(chunk, "to_numpy"):
            chunk = chunk.to_numpy()
        chunk = np.asarray(chunk, dtype=np.float32)
        chunks.append(chunk)
        elapsed = time.time() - t0
        done = i + 1
        rate = (end / elapsed) if elapsed > 0 else 0
        eta = (n - end) / rate if rate > 0 else 0
        logger.info(
            "Transform batch %d/%d (samples %d/%d) | %.1fs elapsed | ETA %.0fs",
            done, n_batches, end, n, elapsed, eta,
        )
    out = np.vstack(chunks)
    logger.info("Transformed shape %s -> %s in %.1fs", X.shape, out.shape, time.time() - t0)
    return out


def save_multirocket_transformer(transformer: Any, path: Union[str, Path]) -> Path:
    """Persist fitted transformer with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, path)
    logger.info("Saved MultiROCKET transformer to %s", path)
    return path


def load_multirocket_transformer(path: Union[str, Path]) -> Any:
    """Load transformer from joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MultiROCKET transformer not found: {path}")
    return joblib.load(path)


def extract_kernel_info(transformer: Any) -> dict:
    """Extract per-kernel metadata from a fitted MultiRocketMultivariate transformer.

    Returns dict with:
        dilations: 1D int array of dilation values (one per dilation level)
        num_features_per_dilation: 1D int array (features per kernel at each dilation)
        n_kernels_per_group: 84 (fixed by MultiRocket's 84 kernel shapes)
        n_base_features: total base features (sum of 84 * num_features_per_dilation)
        n_origins: 2 (raw + differenced)
        n_stats: 4
        stat_names: ["PPV", "LSPV", "MPV", "MIPV"]
        origin_names: ["raw", "differenced"]
    """
    _, _, dilations, num_features_per_dilation, biases = transformer.parameter
    dilations = np.asarray(dilations)
    nfpd = np.asarray(num_features_per_dilation)
    return {
        "dilations": dilations,
        "num_features_per_dilation": nfpd,
        "n_kernels_per_group": 84,
        "n_base_features": len(biases),
        "n_origins": 2,
        "n_stats": 4,
        "stat_names": ["PPV", "LSPV", "MPV", "MIPV"],
        "origin_names": ["raw", "differenced"],
    }
