"""Multiclass MLP classifier with CrossEntropyLoss."""

import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLPClassifier(nn.Module):
    """MLP for multiclass classification: linear layers with ReLU and dropout, then logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
