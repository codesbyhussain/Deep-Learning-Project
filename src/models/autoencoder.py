"""Feedforward autoencoder for feature reduction; configurable hidden dims and latent dim."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class FeedforwardAE(nn.Module):
    """Encoder: input -> hidden layers -> latent. Decoder: latent -> hidden layers -> output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder
        enc_layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder (reverse hidden dims)
        dec_layers: List[nn.Module] = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def train_autoencoder(
    X_train: torch.Tensor,
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, nn.Module]:
    """
    Train feedforward AE; return (encoder_module, full_ae_module).
    encoder_module is the encoder only (for reducing features); full_ae_module is the full AE for checkpointing.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = FeedforwardAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    n = len(X_train)
    ae.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=X_train.device)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = X_train[idx].to(device)
            recon, _ = ae(batch)
            loss = nn.functional.mse_loss(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info("AE epoch %d loss %.6f", ep + 1, total_loss / max(n_batches, 1))
    return ae.encoder, ae


def save_autoencoder(encoder: nn.Module, path: Path) -> Path:
    """Save encoder state_dict for later use as feature reducer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), path)
    logger.info("Saved autoencoder to %s", path)
    return path


def load_encoder_for_inference(
    path: Path,
    input_dim: int,
    hidden_dims: List[int],
    latent_dim: int,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load encoder state_dict into a FeedforwardAE and return its encoder only."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae = FeedforwardAE(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim)
    state = torch.load(path, map_location=device)
    ae.encoder.load_state_dict(state, strict=False)
    return ae.encoder
