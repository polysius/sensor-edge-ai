import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sensor_edge.models import CNN1DRegressor, LSTMRegressor
from sensor_edge.train import TrainConfig, predict_rul, train_model

N_SENSORS = 4
WINDOW = 10
RUL_CAP = 50


def make_learnable_windows(n_windows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Windows whose last reading carries the target, so any working model can learn it."""
    rng = np.random.default_rng(seed)
    rul = rng.uniform(0, RUL_CAP, size=n_windows)
    windows = rng.normal(0, 0.1, size=(n_windows, WINDOW, N_SENSORS))
    windows[:, :, 0] += (rul / RUL_CAP)[:, None]
    return windows.astype(np.float32), rul


@pytest.mark.parametrize(
    "model",
    [LSTMRegressor(N_SENSORS), CNN1DRegressor(N_SENSORS, WINDOW)],
    ids=["lstm", "cnn"],
)
def test_model_outputs_one_value_per_window(model):
    windows = torch.zeros(7, WINDOW, N_SENSORS)

    assert model(windows).shape == (7,)


@pytest.mark.parametrize(
    "make_model",
    [lambda: LSTMRegressor(N_SENSORS, hidden_size=16), lambda: CNN1DRegressor(N_SENSORS, WINDOW)],
    ids=["lstm", "cnn"],
)
def test_training_beats_predicting_the_mean(make_model):
    train_windows, train_rul = make_learnable_windows(600, seed=0)
    val_windows, val_rul = make_learnable_windows(200, seed=1)
    mean_guess_rmse = float(np.sqrt(np.mean((val_rul - train_rul.mean()) ** 2)))

    result = train_model(
        make_model(), train_windows, train_rul, val_windows, val_rul, RUL_CAP,
        TrainConfig(epochs=15, batch_size=64, learning_rate=3e-3),
    )

    assert result.val_rmse < 0.5 * mean_guess_rmse


def test_training_is_reproducible_for_a_seed():
    train_windows, train_rul = make_learnable_windows(300, seed=0)
    val_windows, val_rul = make_learnable_windows(100, seed=1)
    config = TrainConfig(epochs=3, batch_size=64, seed=5)

    def run():
        torch.manual_seed(config.seed)  # initial weights are drawn when the model is built
        return train_model(
            CNN1DRegressor(N_SENSORS, WINDOW), train_windows, train_rul, val_windows, val_rul,
            RUL_CAP, config,
        )

    first, second = run(), run()

    assert first.val_rmse == second.val_rmse


def test_early_stopping_restores_the_best_epoch():
    train_windows, train_rul = make_learnable_windows(300, seed=0)
    val_windows, val_rul = make_learnable_windows(100, seed=1)

    result = train_model(
        LSTMRegressor(N_SENSORS, hidden_size=8), train_windows, train_rul, val_windows, val_rul,
        RUL_CAP, TrainConfig(epochs=30, patience=2, learning_rate=1e-2),
    )

    from sensor_edge.metrics import rmse

    restored_rmse = rmse(val_rul, predict_rul(result.model, val_windows, RUL_CAP))
    assert restored_rmse == pytest.approx(result.val_rmse)


def test_run_deep_models_reports_mean_and_spread_over_seeds(tmp_path):
    from sensor_edge.deep_experiment import run_deep_models
    from sensor_edge.experiment import ExperimentConfig
    from synthetic import write_subset

    write_subset(tmp_path, "FD001")

    rows = run_deep_models(
        tmp_path, "FD001", ExperimentConfig(window_length=10, rul_cap=30), seeds=[0, 1],
        train_config=TrainConfig(epochs=2, batch_size=64),
    )

    assert [row["model"] for row in rows] == ["LSTM", "1D-CNN"]
    for row in rows:
        assert row["n_seeds"] == 2
        assert np.isfinite([row["test_rmse"], row["test_rmse_std"], row["val_rmse"]]).all()


def test_predictions_are_in_cycles_and_never_negative():
    model = CNN1DRegressor(N_SENSORS, WINDOW)
    windows, _ = make_learnable_windows(20, seed=0)

    predictions = predict_rul(model, windows, RUL_CAP)

    assert predictions.shape == (20,)
    assert (predictions >= 0).all()
