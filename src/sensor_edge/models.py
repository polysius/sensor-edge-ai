"""Neural RUL regressors. Both take windows shaped (batch, window_length, n_sensors)."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Reads the window cycle by cycle and predicts RUL from the last hidden state."""

    def __init__(self, n_sensors: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_sensors,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(windows)
        last_cycle = outputs[:, -1, :]
        return self.head(last_cycle).squeeze(-1)


class CNN1DRegressor(nn.Module):
    """Slides small filters along the time axis, then maps the feature map to RUL."""

    def __init__(self, n_sensors: int, window_length: int, channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_sensors, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, 2 * channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * channels * window_length, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # Conv1d expects (batch, channels, time), so sensors become channels.
        channels_first = windows.transpose(1, 2)
        return self.head(self.features(channels_first)).squeeze(-1)
