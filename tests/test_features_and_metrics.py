import numpy as np
import pytest

from sensor_edge.features import summarise_windows
from sensor_edge.metrics import nasa_score, rmse


def test_summary_features_of_a_linear_ramp():
    ramp = np.arange(10, dtype=float).reshape(1, 10, 1) * 2  # rises by 2 per cycle

    features = summarise_windows(ramp)

    mean, std, slope, last = features[0]
    assert mean == pytest.approx(9.0)
    assert std == pytest.approx(np.arange(10).std() * 2)
    assert slope == pytest.approx(2.0)
    assert last == 18.0


def test_summary_features_have_four_columns_per_sensor():
    windows = np.random.default_rng(0).normal(size=(7, 30, 5))

    assert summarise_windows(windows).shape == (7, 20)


def test_rmse():
    assert rmse(np.array([10.0, 20.0]), np.array([13.0, 16.0])) == pytest.approx(
        np.sqrt((9 + 16) / 2)
    )


def test_nasa_score_is_zero_for_a_perfect_prediction():
    assert nasa_score(np.array([50.0]), np.array([50.0])) == 0.0


def test_nasa_score_values_for_an_early_and_a_late_prediction():
    early = nasa_score(np.array([100.0]), np.array([90.0]))
    late = nasa_score(np.array([100.0]), np.array([110.0]))

    assert early == pytest.approx(np.exp(10 / 13) - 1)
    assert late == pytest.approx(np.exp(10 / 10) - 1)


def test_nasa_score_penalises_late_predictions_more_than_early_ones():
    assert nasa_score(np.array([100.0]), np.array([110.0])) > nasa_score(
        np.array([100.0]), np.array([90.0])
    )


def test_nasa_score_sums_over_units():
    true = np.array([100.0, 100.0])
    predicted = np.array([90.0, 110.0])

    assert nasa_score(true, predicted) == pytest.approx(np.exp(10 / 13) - 1 + np.exp(1) - 1)
