"""Compare PyTorch, ONNX and INT8 versions of the neural RUL models on size, speed and accuracy."""

import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from sensor_edge.data import load_subset
from sensor_edge.deep_experiment import MODEL_FACTORIES
from sensor_edge.edge_bench import (
    create_session,
    latency_percentiles,
    predict_scaled,
    scaled_to_rul,
)
from sensor_edge.experiment import ExperimentConfig, PreparedData, prepare_data
from sensor_edge.metrics import nasa_score, rmse
from sensor_edge.onnx_export import export_onnx
from sensor_edge.quantization import quantize_dynamic_int8, quantize_static_int8
from sensor_edge.train import TrainConfig, predict_rul, train_model

SRC_DIR = Path(__file__).resolve().parents[1]
BYTES_PER_KB = 1024


@dataclass(frozen=True)
class EdgeConfig:
    runs: int = 1000  # timed single-window predictions per variant
    warmup: int = 100
    threads: int = 1  # an edge device is assumed to give the model one core
    calibration_windows: int = 256  # training windows used to set INT8 activation ranges
    seed: int = 0


def _pytorch_latency(model: nn.Module, windows: np.ndarray, edge: EdgeConfig) -> dict:
    torch.set_num_threads(edge.threads)
    inputs = torch.as_tensor(windows, dtype=torch.float32)

    def run_once(i: int) -> None:
        start = i % len(inputs)
        with torch.no_grad():
            model(inputs[start : start + 1])

    return latency_percentiles(run_once, edge.warmup, edge.runs)


def _pytorch_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def _worker_benchmark(model_path: Path, windows_path: Path, edge: EdgeConfig) -> dict:
    """Latency and memory of one ONNX file, measured in a fresh process."""
    env = {**os.environ, "PYTHONPATH": str(SRC_DIR)}
    completed = subprocess.run(
        [
            sys.executable, "-m", "sensor_edge.edge_worker",
            "--model", str(model_path), "--windows", str(windows_path),
            "--warmup", str(edge.warmup), "--runs", str(edge.runs),
            "--threads", str(edge.threads),
        ],
        capture_output=True, text=True, check=True, env=env,
    )
    return json.loads(completed.stdout)


def _evenly_spaced(windows: np.ndarray, count: int) -> np.ndarray:
    indices = np.linspace(0, len(windows) - 1, num=min(count, len(windows))).astype(int)
    return windows[indices]


def _train(name: str, data: PreparedData, experiment: ExperimentConfig, seed: int) -> nn.Module:
    window_length, n_sensors = data.train_windows.shape[1:]
    torch.manual_seed(seed)  # seed before building the model so the initial weights repeat
    model = MODEL_FACTORIES[name](n_sensors, window_length)
    result = train_model(
        model, data.train_windows, data.train_rul, data.val_windows, data.val_rul,
        experiment.rul_cap, TrainConfig(seed=seed),
    )
    return result.model


def run_edge_benchmark(
    data_dir: Path,
    subset_name: str,
    experiment: ExperimentConfig,
    edge: EdgeConfig,
    workdir: Path,
) -> list[dict]:
    """Train each network, build ONNX and INT8 versions, and measure all of them.

    Accuracy is scored on the last window of every official test engine. The INT8
    activation ranges come from training windows, never from test windows.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(load_subset(data_dir, subset_name), experiment)
    window_length, n_sensors = data.train_windows.shape[1:]
    windows_path = workdir / "test_windows.npy"
    np.save(windows_path, data.test_windows.astype(np.float32))
    calibration = _evenly_spaced(data.train_windows, edge.calibration_windows)

    def score(predictions: np.ndarray) -> dict:
        return {
            "test_rmse": rmse(data.test_rul, predictions),
            "test_nasa_score": nasa_score(data.test_rul, predictions),
        }

    rows = []
    for name in MODEL_FACTORIES:
        model = _train(name, data, experiment, edge.seed)
        torch_predictions = predict_rul(model, data.test_windows, experiment.rul_cap)
        torch_score = score(torch_predictions)
        rows.append({
            "model": name, "variant": "PyTorch fp32",
            "size_kb": _pytorch_size_bytes(model) / BYTES_PER_KB,
            **_pytorch_latency(model, data.test_windows, edge),
            "rss_delta_mb": None, **torch_score, "rmse_change": 0.0, "max_diff_cycles": None,
        })

        fp32_path = workdir / f"{name}_fp32.onnx"
        export_onnx(model, window_length, n_sensors, fp32_path)
        dynamic_path = workdir / f"{name}_dynamic_int8.onnx"
        quantize_dynamic_int8(fp32_path, dynamic_path)
        static_path = workdir / f"{name}_static_int8.onnx"
        quantize_static_int8(fp32_path, static_path, calibration)

        variants = {
            "ONNX fp32": fp32_path,
            "ONNX dynamic INT8": dynamic_path,
            "ONNX static INT8": static_path,
        }
        fp32_onnx_predictions = None
        for variant, path in variants.items():
            session = create_session(path, edge.threads)
            predictions = scaled_to_rul(
                predict_scaled(session, data.test_windows), experiment.rul_cap
            )
            if fp32_onnx_predictions is None:
                fp32_onnx_predictions = predictions
            variant_score = score(predictions)
            rows.append({
                "model": name, "variant": variant,
                "size_kb": path.stat().st_size / BYTES_PER_KB,
                **_worker_benchmark(path, windows_path, edge),
                **variant_score,
                "rmse_change": variant_score["test_rmse"] - torch_score["test_rmse"],
                "max_diff_cycles": float(np.abs(predictions - fp32_onnx_predictions).max()),
            })
    return rows
