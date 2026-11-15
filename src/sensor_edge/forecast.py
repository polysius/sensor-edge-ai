"""Forecast when an engine's health indicator reaches the failure level, with an interval."""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

from sensor_edge.data import Subset
from sensor_edge.experiment import ExperimentConfig, normalise_subset
from sensor_edge.health import HealthIndicator
from sensor_edge.metrics import nasa_score, rmse

TIME_SCALE = 100.0  # cycles per unit of kernel time, so length scales stay near 1
N_SIGMA = 2.0  # half-width of the interval in standard deviations, about 95 percent
HORIZON = 250  # cycles ahead to search for the failure level


@dataclass(frozen=True)
class FailureForecast:
    """Forecast for one engine, in cycles of remaining life after its last observed cycle."""

    mean_rul: float
    lower_rul: float  # earliest plausible failure
    upper_rul: float  # latest plausible failure
    future_cycles: np.ndarray
    future_mean: np.ndarray
    future_std: np.ndarray


def _make_gp(seed: int) -> GaussianProcessRegressor:
    """A quadratic trend plus a smooth wiggle plus noise.

    The squared dot product kernel is Bayesian quadratic regression, so the
    uncertainty in the trend itself widens the band as it extrapolates.
    """
    trend = ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(1.0, (1e-2, 1e2)) ** 2
    wiggle = ConstantKernel(0.1, (1e-4, 1e1)) * RBF(0.3, (0.05, 2.0))
    noise = WhiteKernel(0.01, (1e-5, 1.0))
    return GaussianProcessRegressor(trend + wiggle + noise, normalize_y=True, random_state=seed)


def _first_crossing(curve: np.ndarray, level: float) -> float:
    """Cycles ahead (counting from 1) at which `curve` first reaches `level`, else HORIZON."""
    reached = np.flatnonzero(curve >= level)
    return float(reached[0] + 1) if len(reached) else float(HORIZON)


def forecast_failure(
    cycles: np.ndarray, health: np.ndarray, failure_level: float, seed: int = 0
) -> FailureForecast:
    gp = _make_gp(seed)
    with warnings.catch_warnings():
        # Short histories often pin a hyperparameter at its bound. That is acceptable here.
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit((cycles / TIME_SCALE)[:, None], health)

    future_cycles = cycles[-1] + np.arange(1, HORIZON + 1)
    mean, std = gp.predict((future_cycles / TIME_SCALE)[:, None], return_std=True)
    return FailureForecast(
        mean_rul=_first_crossing(mean, failure_level),
        lower_rul=_first_crossing(mean + N_SIGMA * std, failure_level),
        upper_rul=_first_crossing(mean - N_SIGMA * std, failure_level),
        future_cycles=future_cycles,
        future_mean=mean,
        future_std=std,
    )


def forecast_test_engines(
    subset: Subset, config: ExperimentConfig
) -> tuple[pd.DataFrame, HealthIndicator, dict[int, FailureForecast]]:
    """Fit the health indicator on training engines and forecast every test engine.

    Returns a table of forecasts against the official RUL, the fitted indicator,
    and the full forecast per test unit for plotting.
    """
    frames = normalise_subset(subset, config)
    indicator = HealthIndicator(frames.sensors, config.rul_cap).fit(frames.train)

    forecasts, rows = {}, []
    for unit, engine in frames.test.groupby("unit"):
        cycles, health = indicator.trajectory(engine)
        forecast = forecast_failure(cycles, health, indicator.failure_level_, config.seed)
        forecasts[unit] = forecast
        rows.append(
            {
                "unit": unit,
                "true_rul": subset.test_rul.loc[unit],
                "mean_rul": forecast.mean_rul,
                "lower_rul": forecast.lower_rul,
                "upper_rul": forecast.upper_rul,
            }
        )
    return pd.DataFrame(rows), indicator, forecasts


# True RUL ranges for the breakdown. Engines far from failure show no degradation yet.
RUL_BUCKETS = [("0 to 30", 0, 30), ("31 to 60", 31, 60), ("61 to 125", 61, 125), ("over 125", 126, 10_000)]


def summarise_by_true_rul(table: pd.DataFrame) -> list[dict]:
    """Summaries per range of true RUL, which show where a forecast can work at all.

    The ranges use the official test RUL, so this is an analysis of the results and
    not something a deployed system could know in advance.
    """
    rows = []
    for label, low, high in RUL_BUCKETS:
        in_bucket = table[(table["true_rul"] >= low) & (table["true_rul"] <= high)]
        if len(in_bucket):
            rows.append({"true_rul_range": label, **summarise_forecasts(in_bucket)})
    return rows


def summarise_forecasts(table: pd.DataFrame) -> dict:
    """Point error and how often the true RUL falls inside the interval."""
    inside = (table["true_rul"] >= table["lower_rul"]) & (table["true_rul"] <= table["upper_rul"])
    return {
        "rmse": rmse(table["true_rul"].to_numpy(), table["mean_rul"].to_numpy()),
        "nasa_score": nasa_score(table["true_rul"].to_numpy(), table["mean_rul"].to_numpy()),
        "coverage": float(inside.mean()),
        "mean_width": float((table["upper_rul"] - table["lower_rul"]).mean()),
        "censored_share": float((table["upper_rul"] >= HORIZON).mean()),
        "n_engines": len(table),
    }
