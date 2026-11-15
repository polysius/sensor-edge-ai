import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("psutil")

from sensor_edge.edge_bench import (
    create_session,
    latency_percentiles,
    predict_scaled,
    scaled_to_rul,
)
from sensor_edge.models import CNN1DRegressor, LSTMRegressor
from sensor_edge.onnx_export import export_onnx
from sensor_edge.quantization import (
    WindowCalibrationReader,
    quantize_dynamic_int8,
    quantize_static_int8,
)

WINDOW, SENSORS = 30, 14
SRC = Path(__file__).resolve().parents[1] / "src"


def make_windows(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, WINDOW, SENSORS)).astype(np.float32)


MODELS = {
    "lstm": lambda: LSTMRegressor(SENSORS),
    "cnn": lambda: CNN1DRegressor(SENSORS, WINDOW),
}


@pytest.fixture(params=sorted(MODELS))
def exported(request, tmp_path):
    """A model with random weights, its torch outputs on a probe set, and its ONNX file."""
    torch.manual_seed(0)
    model = MODELS[request.param]().eval()
    path = tmp_path / f"{request.param}.onnx"
    export_onnx(model, WINDOW, SENSORS, path)
    probe = make_windows(40, seed=1)
    with torch.no_grad():
        reference = model(torch.as_tensor(probe)).numpy()
    return path, probe, reference


def test_onnx_matches_pytorch_for_any_batch_size(exported):
    path, probe, reference = exported
    session = create_session(path)

    full = predict_scaled(session, probe)
    single = predict_scaled(session, probe[:1])

    assert np.allclose(full, reference, atol=1e-4)
    assert np.allclose(single, reference[:1], atol=1e-4)


def test_dynamic_quantisation_shrinks_the_model_and_stays_close(exported, tmp_path):
    path, probe, reference = exported
    quantised_path = tmp_path / "dynamic.onnx"

    quantize_dynamic_int8(path, quantised_path)

    assert quantised_path.stat().st_size < path.stat().st_size
    outputs = predict_scaled(create_session(quantised_path), probe)
    assert np.isfinite(outputs).all()
    assert np.abs(outputs - reference).max() < 0.1  # outputs are RUL / cap, so 0.1 is 12 cycles


def test_static_quantisation_runs_with_calibration_windows(exported, tmp_path):
    path, probe, reference = exported
    quantised_path = tmp_path / "static.onnx"

    quantize_static_int8(path, quantised_path, make_windows(64, seed=2))

    outputs = predict_scaled(create_session(quantised_path), probe)
    assert outputs.shape == reference.shape
    assert np.isfinite(outputs).all()
    assert np.abs(outputs - reference).max() < 0.1


def op_types(path):
    import onnx

    return {node.op_type for node in onnx.load(str(path)).graph.node}


def test_dynamic_quantisation_covers_convolutions_and_lstm(exported, tmp_path):
    path, _, _ = exported
    quantised_path = tmp_path / "dynamic.onnx"

    quantize_dynamic_int8(path, quantised_path)

    quantised_ops = op_types(quantised_path)
    if "LSTM" in op_types(path):
        assert "DynamicQuantizeLSTM" in quantised_ops and "LSTM" not in quantised_ops
    else:
        assert "ConvInteger" in quantised_ops and "Conv" not in quantised_ops
    assert "MatMulInteger" in quantised_ops


def test_worker_does_not_import_torch():
    code = (
        "import sys; import sensor_edge.edge_worker; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], check=False, env={"PYTHONPATH": str(SRC), "PATH": ""}
    )

    assert completed.returncode == 0


def test_calibration_reader_yields_every_window_once_then_stops():
    reader = WindowCalibrationReader(make_windows(70), batch_size=32)

    sizes = []
    while (batch := reader.get_next()) is not None:
        sizes.append(len(batch["window"]))

    assert sizes == [32, 32, 6]


def test_latency_percentiles_orders_p50_below_p95_and_counts_calls():
    calls = []

    result = latency_percentiles(lambda i: calls.append(i), n_warmup=5, n_runs=50)

    assert len(calls) == 55
    assert 0 <= result["p50_ms"] <= result["p95_ms"]


def test_scaled_output_converts_to_cycles_and_never_goes_negative():
    cycles = scaled_to_rul(np.array([-0.1, 0.0, 0.5, 1.0]), rul_cap=125)

    assert cycles.tolist() == [0.0, 0.0, 62.5, 125.0]


def test_worker_reports_latency_and_memory_for_a_model(exported, tmp_path):
    path, probe, _ = exported
    windows_path = tmp_path / "windows.npy"
    np.save(windows_path, probe)

    completed = subprocess.run(
        [sys.executable, "-m", "sensor_edge.edge_worker", "--model", str(path),
         "--windows", str(windows_path), "--warmup", "5", "--runs", "30"],
        capture_output=True, text=True, check=True,
        env={"PYTHONPATH": str(SRC), "PATH": ""},
    )

    result = json.loads(completed.stdout)
    assert 0 < result["p50_ms"] <= result["p95_ms"]
    assert result["rss_delta_mb"] >= 0
