"""Autoencoder anomaly detector. A window it cannot rebuild well is unlike healthy data."""

import numpy as np
import torch
from torch import nn


class _WindowAutoencoder(nn.Module):
    """Squeezes a flattened window through a small bottleneck and rebuilds it."""

    def __init__(self, n_inputs: int, hidden_size: int = 128, bottleneck_size: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_inputs),
        )

    def forward(self, flat_windows: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(flat_windows))


class AutoencoderDetector:
    """Scores a window by its mean squared reconstruction error."""

    def __init__(self, seed: int = 0, epochs: int = 30, batch_size: int = 256, learning_rate: float = 1e-3):
        self.seed = seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._network: _WindowAutoencoder | None = None

    def fit(self, healthy_windows: np.ndarray) -> "AutoencoderDetector":
        torch.manual_seed(self.seed)
        flat = torch.as_tensor(healthy_windows.reshape(len(healthy_windows), -1), dtype=torch.float32)
        self._network = _WindowAutoencoder(flat.shape[1])
        optimiser = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        shuffle_rng = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            flat, batch_size=self.batch_size, shuffle=True, generator=shuffle_rng
        )
        self._network.train()
        for _ in range(self.epochs):
            for batch in loader:
                optimiser.zero_grad()
                nn.functional.mse_loss(self._network(batch), batch).backward()
                optimiser.step()
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("Call fit before score")
        self._network.eval()
        flat = torch.as_tensor(windows.reshape(len(windows), -1), dtype=torch.float32)
        with torch.no_grad():
            rebuilt = self._network(flat)
        return ((rebuilt - flat) ** 2).mean(dim=1).numpy()
