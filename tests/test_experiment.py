import numpy as np
import pytest

from sensor_edge.data import load_subset
from sensor_edge.experiment import ExperimentConfig, prepare_data, run_baselines
from synthetic import write_subset

CONFIG = ExperimentConfig(window_length=10, rul_cap=30)


@pytest.fixture
def data_dir(tmp_path):
    write_subset(tmp_path, "FD001")
    return tmp_path


def test_train_and_validation_windows_come_from_different_units(data_dir):
    data = prepare_data(load_subset(data_dir, "FD001"), CONFIG)

    assert set(data.train_units).isdisjoint(data.val_units)


def test_test_set_has_one_window_per_unit_with_matching_official_rul(data_dir):
    subset = load_subset(data_dir, "FD001")

    data = prepare_data(subset, CONFIG)

    assert data.test_windows.shape[0] == subset.test["unit"].nunique()
    assert np.array_equal(data.test_rul, subset.test_rul.to_numpy())
    assert data.test_windows.shape[1:] == (CONFIG.window_length, len(data.sensors))


def test_constant_sensor_is_not_used(data_dir):
    data = prepare_data(load_subset(data_dir, "FD001"), CONFIG)

    assert "sensor_1" not in data.sensors


def test_training_target_is_capped(data_dir):
    data = prepare_data(load_subset(data_dir, "FD001"), CONFIG)

    assert data.train_rul.max() == CONFIG.rul_cap
    assert data.train_rul.min() == 0


@pytest.mark.parametrize("bad_config", [{"window_length": 1}, {"val_fraction": 1.0}, {"val_fraction": 0}])
def test_invalid_config_is_rejected(bad_config):
    with pytest.raises(ValueError):
        ExperimentConfig(**bad_config)


def test_run_baselines_reports_finite_metrics_for_both_models(data_dir):
    rows = run_baselines(data_dir, "FD001", CONFIG)

    assert [row["model"] for row in rows] == ["Linear regression", "Gradient boosting"]
    for row in rows:
        assert np.isfinite([row["val_rmse"], row["test_rmse"], row["test_nasa_score"]]).all()


def test_a_model_that_sees_degradation_beats_predicting_the_mean(data_dir):
    rows = run_baselines(data_dir, "FD001", CONFIG)
    data = prepare_data(load_subset(data_dir, "FD001"), CONFIG)
    mean_guess_rmse = float(np.sqrt(np.mean((data.val_rul - data.train_rul.mean()) ** 2)))

    assert min(row["val_rmse"] for row in rows) < mean_guess_rmse
