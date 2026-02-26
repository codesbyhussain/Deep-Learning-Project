"""Training callbacks (e.g. early stopping)."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop when validation metric does not improve for patience epochs."""

    def __init__(
        self,
        patience: int = 5,
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.patience = patience
        self.mode = mode  # "max" for accuracy/F1, "min" for loss
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """Record value; return True if should stop."""
        if self.best is None:
            self.best = value
            return False
        if self.mode == "max":
            improved = value > self.best and (value - self.best) >= self.min_delta
        else:
            improved = value < self.best and (self.best - value) >= self.min_delta
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info("Early stopping triggered after %d epochs without improvement", self.patience)
            return True
        return False
