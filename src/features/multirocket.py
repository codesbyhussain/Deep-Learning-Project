"""MultiROCKET wrapper: use existing package (sktime preferred); fit on train, batch transform, persist."""

import logging
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
    """Create MultiRocket transformer from sktime if available. Uses MultiRocketMultivariate (num_kernels rounded to multiple of 84)."""
    try:
        from sktime.transformations.panel.rocket import MultiRocketMultivariate

        return MultiRocketMultivariate(num_kernels=num_kernels, random_state=seed)
    except ImportError:
        try:
            from sktime.transformations.panel.rocket import MultiRocket

            return MultiRocket(n_kernels=num_kernels, random_state=seed)
        except ImportError:
            return None


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
    if trans is None:
        raise ImportError(
            "MultiROCKET requires sktime. Install with: pip install sktime. "
            "Or pass a custom adapter to create_multirocket_transformer(adapter=...)."
        )
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
    """Transform in batches; return (n_samples, n_features) float array."""
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    n = len(X)
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = transformer.transform(X[start:end])
        if hasattr(chunk, "to_numpy"):
            chunk = chunk.to_numpy()
        chunk = np.asarray(chunk, dtype=np.float32)
        chunks.append(chunk)
    out = np.vstack(chunks)
    logger.debug("Transformed shape %s -> %s", X.shape, out.shape)
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
