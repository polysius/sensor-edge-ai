"""Latency measurement and ONNX inference helpers."""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from sensor_edge.onnx_io import INPUT_NAME


def latency_percentiles(run_once: Callable[[int], object], n_warmup: int, n_runs: int) -> dict:
    """Time `run_once(i)` for `n_runs` calls after `n_warmup` untimed ones.

    The index lets the caller cycle through different inputs. Returns milliseconds.
    """
    for i in range(n_warmup):
        run_once(i)
    durations = np.empty(n_runs)
    for i in range(n_runs):
        started = time.perf_counter()
        run_once(i)
        durations[i] = time.perf_counter() - started
    milliseconds = durations * 1000
    return {
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
    }


def create_session(model_path: Path, threads: int = 1):
    """An ONNX Runtime CPU session. One thread by default, as on a small edge device."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])


def predict_scaled(session, windows: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run `windows` through the session and return the raw network output (RUL / cap)."""
    outputs = [
        session.run(None, {INPUT_NAME: windows[start : start + batch_size].astype(np.float32)})[0]
        for start in range(0, len(windows), batch_size)
    ]
    return np.concatenate(outputs)


def scaled_to_rul(scaled: np.ndarray, rul_cap: int) -> np.ndarray:
    """Convert network output to cycles of remaining life, never below zero."""
    return np.clip(scaled * rul_cap, 0, None)
