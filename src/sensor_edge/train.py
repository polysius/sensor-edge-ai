"""Train and run neural RUL models."""

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sensor_edge.metrics import rmse


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 8  # stop after this many epochs without a better validation RMSE
    seed: int = 0


@dataclass(frozen=True)
class TrainResult:
    model: nn.Module  # weights from the epoch with the best validation RMSE
    best_epoch: int
    val_rmse: float
    seconds: float


def predict_rul(model: nn.Module, windows: np.ndarray, rul_cap: int, batch_size: int = 1024) -> np.ndarray:
    """Predict RUL in cycles. The network outputs RUL divided by the cap."""
    model.eval()
    inputs = torch.as_tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        scaled = [model(inputs[i : i + batch_size]) for i in range(0, len(inputs), batch_size)]
    return np.clip(torch.cat(scaled).numpy() * rul_cap, 0, None)


def train_model(
    model: nn.Module,
    train_windows: np.ndarray,
    train_rul: np.ndarray,
    val_windows: np.ndarray,
    val_rul: np.ndarray,
    rul_cap: int,
    config: TrainConfig,
) -> TrainResult:
    """Fit with Adam and MSE, keeping the weights that score best on validation engines.

    The seed controls batch order and dropout. Initial weights are drawn when the
    model is built, so seed torch before constructing it to make a run repeatable.
    """
    started = time.perf_counter()
    torch.manual_seed(config.seed)
    shuffle_rng = torch.Generator().manual_seed(config.seed)

    inputs = torch.as_tensor(train_windows, dtype=torch.float32)
    # Scaling the target to [0, 1] keeps the loss on a similar scale for both models.
    targets = torch.as_tensor(train_rul / rul_cap, dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=shuffle_rng,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()

    best_rmse, best_epoch, best_weights = float("inf"), 0, copy.deepcopy(model.state_dict())
    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_inputs, batch_targets in loader:
            optimiser.zero_grad()
            loss_function(model(batch_inputs), batch_targets).backward()
            optimiser.step()

        val_rmse = rmse(val_rul, predict_rul(model, val_windows, rul_cap))
        if val_rmse < best_rmse:
            best_rmse, best_epoch = val_rmse, epoch
            best_weights = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_weights)
    return TrainResult(model, best_epoch, best_rmse, time.perf_counter() - started)
