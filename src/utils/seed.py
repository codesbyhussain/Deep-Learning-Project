"""Reproducibility: set global seeds for Python, NumPy, and PyTorch."""

import logging
import os
import random
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Project-wide default seed. Use this everywhere (splits, MultiROCKET, training) unless overridden.
SEED = 0


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch; enable deterministic behavior where possible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic algorithms when available (may be slower)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info("Global seed set to %s", seed)
