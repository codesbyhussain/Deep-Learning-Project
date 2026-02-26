"""Lightweight FT-Transformer-style model: feature tokenizer + [CLS] + Transformer + classification head."""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeatureTokenizer(nn.Module):
    """Project each feature to d_token; output (batch, n_features + 1, d_token) with [CLS] prepended."""

    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        self.proj = nn.Linear(1, d_token)  # each feature is 1-dim
        self.cls_embed = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        batch = x.shape[0]
        xx = x.unsqueeze(-1)  # (batch, n_features, 1)
        tokens = self.proj(xx)  # (batch, n_features, d_token)
        cls_tok = self.cls_embed.expand(batch, -1, -1)
        return torch.cat([cls_tok, tokens], dim=1)  # (batch, 1 + n_features, d_token)


class FTTransformer(nn.Module):
    """FT-Transformer: tokenize features, Transformer encoder, classify from [CLS]."""

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        d_token: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_ff = d_ff or (d_token * 4)
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_token, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        tokens = self.tokenizer(x)  # (batch, 1 + n_features, d_token)
        encoded = self.transformer(tokens)  # (batch, 1 + n_features, d_token)
        cls_out = encoded[:, 0, :]  # (batch, d_token)
        return self.head(cls_out)
