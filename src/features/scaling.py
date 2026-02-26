"""Feature scaling: fit on train, transform, persist via joblib."""

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on training features. X can be 2D (n_samples, n_features)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("Fitted StandardScaler on shape %s", X_train.shape)
    return scaler


def transform_with_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Transform features with fitted scaler. Returns float64 by default; cast if needed."""
    return scaler.transform(X)


def save_scaler(scaler: StandardScaler, path: Union[str, Path]) -> Path:
    """Persist scaler with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Saved scaler to %s", path)
    return path


def load_scaler(path: Union[str, Path]) -> StandardScaler:
    """Load scaler from joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)
