"""Train and save everything the streaming consumer needs. Needs torch and the ONNX tools."""

from pathlib import Path

import numpy as np
import torch

from sensor_edge.data import load_subset
from sensor_edge.detectors import IsolationForestDetector
from sensor_edge.experiment import ExperimentConfig, normalise_subset
from sensor_edge.models import CNN1DRegressor
from sensor_edge.onnx_export import export_onnx
from sensor_edge.quantization import quantize_static_int8
from sensor_edge.stream_models import ONNX_FILE, StreamBundle, save_bundle
from sensor_edge.train import TrainConfig, train_model
from sensor_edge.windows import make_training_windows

ALARM_QUANTILE = 0.95  # same rule as the M3 detectors: 5 percent of healthy windows may alarm
FOREST_TREES = 100  # fewer trees than M3, because every streamed reading is scored one at a time
CALIBRATION_WINDOWS = 256


def prepare_stream_bundle(
    data_dir: Path,
    subset_name: str,
    experiment: ExperimentConfig,
    out_dir: Path,
    train_config: TrainConfig | None = None,
) -> StreamBundle:
    """Fit the preprocessing, an INT8 1D-CNN for RUL, and an isolation forest on training engines.

    The alarm threshold comes from healthy windows of the validation engines. Test
    engines are only replayed through the stream, never used to fit or choose anything.
    """
    train_config = train_config or TrainConfig(seed=experiment.seed)
    frames = normalise_subset(load_subset(data_dir, subset_name), experiment)

    def windows_with_rul(frame):
        windows, rul, _ = make_training_windows(frame, frames.sensors, experiment.window_length)
        return windows, rul

    train_windows, train_rul = windows_with_rul(frames.train)
    val_windows, val_rul = windows_with_rul(frames.val)
    window_length, n_sensors = train_windows.shape[1:]

    torch.manual_seed(train_config.seed)  # before building the model, so initial weights repeat
    network = CNN1DRegressor(n_sensors, window_length)
    trained = train_model(
        network, train_windows, train_rul, val_windows, val_rul, experiment.rul_cap, train_config
    ).model

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = out_dir / "rul_model_fp32.onnx"
    export_onnx(trained, window_length, n_sensors, fp32_path)
    calibration = train_windows[
        np.linspace(0, len(train_windows) - 1, CALIBRATION_WINDOWS).astype(int)
    ]
    onnx_path = out_dir / ONNX_FILE
    quantize_static_int8(fp32_path, onnx_path, calibration)

    healthy_train = train_windows[train_rul >= experiment.rul_cap]
    healthy_val = val_windows[val_rul >= experiment.rul_cap]
    forest = IsolationForestDetector(seed=experiment.seed, n_estimators=FOREST_TREES)
    forest.fit(healthy_train)
    threshold = float(np.quantile(forest.score(healthy_val), ALARM_QUANTILE))

    bundle = StreamBundle(
        normaliser=frames.normaliser,
        forest=forest,
        sensors=frames.sensors,
        window_length=window_length,
        rul_cap=experiment.rul_cap,
        alarm_threshold=threshold,
        onnx_path=onnx_path,
    )
    save_bundle(bundle, out_dir)
    return bundle
