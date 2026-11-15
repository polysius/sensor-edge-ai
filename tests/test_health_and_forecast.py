import numpy as np
import pandas as pd
import pytest

from sensor_edge.forecast import (
    HORIZON,
    forecast_failure,
    summarise_by_true_rul,
    summarise_forecasts,
)
from sensor_edge.health import HealthIndicator
from sensor_edge.preprocess import OperatingConditionNormaliser, add_capped_rul
from synthetic import make_fleet

SENSORS = [f"sensor_{i}" for i in range(2, 22)]


@pytest.fixture
def training_frame():
    fleet = make_fleet(n_units=15, seed=0)
    normalised = OperatingConditionNormaliser(n_conditions=1).fit(fleet).transform(fleet)
    return add_capped_rul(normalised, cap=30)


def test_health_indicator_rises_as_an_engine_wears(training_frame):
    indicator = HealthIndicator(SENSORS, rul_cap=30).fit(training_frame)

    for _, engine in training_frame.groupby("unit"):
        _, health = indicator.trajectory(engine)
        assert health[-10:].mean() > health[:10].mean()


def test_failure_level_is_above_the_healthy_level(training_frame):
    indicator = HealthIndicator(SENSORS, rul_cap=30).fit(training_frame)

    healthy_rows = training_frame[training_frame["rul"] >= 30]
    healthy_level = np.mean(
        [indicator.trajectory(e)[1].mean() for _, e in healthy_rows.groupby("unit")]
    )
    assert indicator.failure_level_ > healthy_level


def test_smoothing_never_looks_at_later_cycles(training_frame):
    indicator = HealthIndicator(SENSORS, rul_cap=30).fit(training_frame)
    engine = training_frame[training_frame["unit"] == 1]
    altered = engine.copy()
    altered.loc[altered.index[-1], SENSORS] += 100.0

    _, original = indicator.trajectory(engine)
    _, changed = indicator.trajectory(altered)

    assert np.array_equal(original[:-1], changed[:-1])


def quadratic_history(n_cycles: int = 100, seed: int = 0):
    cycles = np.arange(1, n_cycles + 1)
    noise = np.random.default_rng(seed).normal(0, 0.05, size=n_cycles)
    return cycles, 0.0002 * cycles**2 + noise


def test_forecast_finds_the_crossing_of_a_clean_quadratic():
    cycles, health = quadratic_history()
    failure_level = 0.0002 * 160**2  # reached at cycle 160, 60 cycles after the history ends

    forecast = forecast_failure(cycles, health, failure_level)

    assert forecast.mean_rul == pytest.approx(60, abs=15)
    assert forecast.lower_rul <= forecast.mean_rul <= forecast.upper_rul


def test_band_widens_further_from_the_data():
    cycles, health = quadratic_history()

    forecast = forecast_failure(cycles, health, failure_level=100.0)

    assert forecast.future_std[-1] > forecast.future_std[0]


def test_a_level_already_passed_gives_one_cycle_left():
    cycles, health = quadratic_history()

    forecast = forecast_failure(cycles, health, failure_level=health[-1] - 1.0)

    assert forecast.mean_rul == 1


def test_a_level_never_reached_is_reported_at_the_horizon():
    cycles = np.arange(1, 81)
    health = np.random.default_rng(0).normal(0, 0.05, size=80)

    forecast = forecast_failure(cycles, health, failure_level=50.0)

    assert forecast.mean_rul == HORIZON
    assert forecast.upper_rul == HORIZON


def test_breakdown_groups_engines_by_true_rul_and_skips_empty_ranges():
    table = pd.DataFrame(
        {
            "true_rul": [10, 25, 40, 100],
            "mean_rul": [12, 25, 50, 90],
            "lower_rul": [5, 20, 45, 80],
            "upper_rul": [15, 30, 60, 120],
        }
    )

    rows = summarise_by_true_rul(table)

    assert [row["true_rul_range"] for row in rows] == ["0 to 30", "31 to 60", "61 to 125"]
    assert [row["n_engines"] for row in rows] == [2, 1, 1]
    assert rows[0]["coverage"] == 1.0
    assert rows[1]["coverage"] == 0.0  # true RUL 40 is below the lower bound of 45


def test_summary_counts_coverage_and_errors():
    table = pd.DataFrame(
        {
            "true_rul": [50, 80, 20],
            "mean_rul": [55, 70, 20],
            "lower_rul": [40, 60, 25],  # the third interval misses the true value
            "upper_rul": [70, 90, 40],
        }
    )

    summary = summarise_forecasts(table)

    assert summary["coverage"] == pytest.approx(2 / 3)
    assert summary["rmse"] == pytest.approx(np.sqrt((25 + 100 + 0) / 3))
    assert summary["mean_width"] == pytest.approx((30 + 30 + 15) / 3)
