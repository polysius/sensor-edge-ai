"""Train the neural RUL models on one subset over several seeds."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sensor_edge.data import load_subset
from sensor_edge.experiment import ExperimentConfig, PreparedData, prepare_data
from sensor_edge.metrics import nasa_score, rmse
from sensor_edge.models import CNN1DRegressor, LSTMRegressor
from sensor_edge.train import TrainConfig, predict_rul, train_model

ModelFactory = Callable[[int, int], nn.Module]  # (n_sensors, window_length) -> model

MODEL_FACTORIES: dict[str, ModelFactory] = {
    "LSTM": lambda n_sensors, window_length: LSTMRegressor(n_sensors),
    "1D-CNN": lambda n_sensors, window_length: CNN1DRegressor(n_sensors, window_length),
}


def run_one_seed(
    data: PreparedData,
    factory: ModelFactory,
    experiment: ExperimentConfig,
    train_config: TrainConfig,
) -> dict:
    window_length, n_sensors = data.train_windows.shape[1:]
    # Seed before building the model, otherwise the initial weights ignore the seed.
    torch.manual_seed(train_config.seed)
    result = train_model(
        factory(n_sensors, window_length),
        data.train_windows,
        data.train_rul,
        data.val_windows,
        data.val_rul,
        experiment.rul_cap,
        train_config,
    )
    test_predictions = predict_rul(result.model, data.test_windows, experiment.rul_cap)
    return {
        "val_rmse": result.val_rmse,
        "test_rmse": rmse(data.test_rul, test_predictions),
        "test_nasa_score": nasa_score(data.test_rul, test_predictions),
        "best_epoch": result.best_epoch,
        "seconds": result.seconds,
    }


def run_deep_models(
    data_dir: Path,
    subset_name: str,
    experiment: ExperimentConfig,
    seeds: list[int],
    train_config: TrainConfig | None = None,
) -> list[dict]:
    """Train each network once per seed and report the mean and spread across seeds.

    The engine split stays fixed (experiment.seed) so the seeds measure training
    randomness only: weight initialisation and batch order.
    """
    train_config = train_config or TrainConfig()
    data = prepare_data(load_subset(data_dir, subset_name), experiment)
    rows = []
    for name, factory in MODEL_FACTORIES.items():
        runs = [
            run_one_seed(data, factory, experiment, replace(train_config, seed=seed))
            for seed in seeds
        ]
        row = {"subset": subset_name, "model": name, "n_seeds": len(seeds)}
        for metric in ("val_rmse", "test_rmse", "test_nasa_score", "seconds"):
            values = np.array([run[metric] for run in runs])
            row[metric] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    return rows
