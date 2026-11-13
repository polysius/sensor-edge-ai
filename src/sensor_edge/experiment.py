"""Prepare one C-MAPSS subset and evaluate RUL models on it."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensor_edge.baselines import make_baselines
from sensor_edge.data import OPERATING_CONDITIONS, Subset, load_subset, split_units
from sensor_edge.features import summarise_windows
from sensor_edge.metrics import nasa_score, rmse
from sensor_edge.preprocess import (
    OperatingConditionNormaliser,
    add_capped_rul,
    select_informative_sensors,
)
from sensor_edge.windows import make_last_windows, make_training_windows


@dataclass(frozen=True)
class ExperimentConfig:
    rul_cap: int = 125
    window_length: int = 30
    val_fraction: float = 0.2
    seed: int = 0
    n_conditions: int | None = None  # None uses the subset's real number of conditions

    def __post_init__(self):
        if self.window_length < 2:
            raise ValueError("window_length must be at least 2 to compute a slope")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be between 0 and 1")


@dataclass(frozen=True)
class PreparedData:
    """Windowed arrays for one subset. Windows have shape (n, window_length, n_sensors)."""

    sensors: list[str]
    train_windows: np.ndarray
    train_rul: np.ndarray
    train_units: np.ndarray
    val_windows: np.ndarray
    val_rul: np.ndarray
    val_units: np.ndarray
    test_windows: np.ndarray
    test_rul: np.ndarray  # official RUL after each test unit's last cycle
    test_units: np.ndarray


def prepare_data(subset: Subset, config: ExperimentConfig) -> PreparedData:
    train_units, val_units = split_units(
        subset.train["unit"].to_numpy(), config.val_fraction, config.seed
    )
    train = subset.train[subset.train["unit"].isin(train_units)]
    val = subset.train[subset.train["unit"].isin(val_units)]

    n_conditions = config.n_conditions or OPERATING_CONDITIONS[subset.name]
    normaliser = OperatingConditionNormaliser(n_conditions, config.seed).fit(train)
    train, val, test = (normaliser.transform(part) for part in (train, val, subset.test))
    sensors = select_informative_sensors(train)

    train, val = add_capped_rul(train, config.rul_cap), add_capped_rul(val, config.rul_cap)
    train_windows, train_rul, train_window_units = make_training_windows(
        train, sensors, config.window_length
    )
    val_windows, val_rul, val_window_units = make_training_windows(
        val, sensors, config.window_length
    )
    test_windows, test_units = make_last_windows(test, sensors, config.window_length)
    return PreparedData(
        sensors=sensors,
        train_windows=train_windows,
        train_rul=train_rul,
        train_units=train_window_units,
        val_windows=val_windows,
        val_rul=val_rul,
        val_units=val_window_units,
        test_windows=test_windows,
        test_rul=subset.test_rul.loc[test_units].to_numpy(),
        test_units=test_units,
    )


def run_baselines(data_dir: Path, subset_name: str, config: ExperimentConfig) -> list[dict]:
    """Fit each baseline on the training units and score it on validation and test."""
    data = prepare_data(load_subset(data_dir, subset_name), config)
    train_features = summarise_windows(data.train_windows)
    val_features = summarise_windows(data.val_windows)
    test_features = summarise_windows(data.test_windows)

    rows = []
    for name, model in make_baselines(config.seed).items():
        model.fit(train_features, data.train_rul)
        test_predictions = np.clip(model.predict(test_features), 0, None)
        val_predictions = np.clip(model.predict(val_features), 0, None)
        rows.append(
            {
                "subset": subset_name,
                "model": name,
                "val_rmse": rmse(data.val_rul, val_predictions),
                "test_rmse": rmse(data.test_rul, test_predictions),
                "test_nasa_score": nasa_score(data.test_rul, test_predictions),
            }
        )
    return rows
