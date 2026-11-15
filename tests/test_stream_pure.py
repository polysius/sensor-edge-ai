"""Streaming tests that need neither Kafka nor torch."""

import joblib
import numpy as np
import pytest

from sensor_edge.detectors import IsolationForestDetector
from sensor_edge.preprocess import OperatingConditionNormaliser
from sensor_edge.stream_consumer import latency_summary
from sensor_edge.stream_models import ONNX_FILE, StreamBundle, load_bundle, save_bundle
from sensor_edge.stream_producer import SpikeFault, due_time, readings_from_frame
from synthetic import make_fleet


@pytest.fixture
def fleet():
    return make_fleet(n_units=3, seed=0)


def test_readings_are_sent_one_cycle_at_a_time_across_engines(fleet):
    readings = list(readings_from_frame(fleet, faults=[]))

    cycles_and_units = [(r["cycle"], r["unit"]) for r in readings]
    assert cycles_and_units == sorted(cycles_and_units)
    assert len(readings) == len(fleet)
    assert cycles_and_units[:3] == [(1, 1), (1, 2), (1, 3)]


def test_each_reading_carries_every_column_with_plain_types(fleet):
    reading = next(iter(readings_from_frame(fleet, faults=[])))

    assert isinstance(reading["unit"], int) and isinstance(reading["cycle"], int)
    assert {"setting_1", "setting_3", "sensor_1", "sensor_21"} <= reading.keys()


def test_a_spike_changes_only_its_engine_cycles_and_sensor(fleet):
    fault = SpikeFault(unit=2, first_cycle=10, n_cycles=3, sensor="sensor_5", size_in_std=8.0)
    clean = {(r["unit"], r["cycle"]): r for r in readings_from_frame(fleet, faults=[])}

    faulted = list(readings_from_frame(fleet, faults=[fault]))

    expected_jump = 8.0 * fleet["sensor_5"].std()
    for reading in faulted:
        before = clean[(reading["unit"], reading["cycle"])]
        in_fault = reading["unit"] == 2 and 10 <= reading["cycle"] < 13
        for column in ("sensor_5", "sensor_6", "setting_1"):
            jump = reading[column] - before[column]
            if in_fault and column == "sensor_5":
                assert jump == pytest.approx(expected_jump)
            else:
                assert jump == 0


def test_due_time_spaces_messages_evenly():
    assert due_time(start=100.0, index=0, rate=50) == 100.0
    assert due_time(start=100.0, index=25, rate=50) == pytest.approx(100.5)


def test_latency_summary_reports_percentiles_in_milliseconds_and_throughput():
    latencies = [0.001 * i for i in range(1, 101)]  # 1 ms to 100 ms

    summary = latency_summary(latencies, duration_s=2.0)

    assert summary["messages"] == 100
    assert summary["throughput_per_s"] == pytest.approx(50)
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert summary["max_ms"] == pytest.approx(100)


def make_bundle(folder, onnx_bytes=b"not a real model"):
    fleet = make_fleet(n_units=6, seed=0)
    normaliser = OperatingConditionNormaliser(n_conditions=1).fit(fleet)
    sensors = [f"sensor_{i}" for i in range(2, 22)]
    windows = np.random.default_rng(0).normal(size=(200, 10, len(sensors)))
    forest = IsolationForestDetector(seed=0, n_estimators=20).fit(windows)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ONNX_FILE).write_bytes(onnx_bytes)
    return StreamBundle(normaliser, forest, sensors, 10, 30, 0.42, folder / ONNX_FILE), windows


def test_bundle_round_trips_through_disk(tmp_path):
    bundle, windows = make_bundle(tmp_path)

    save_bundle(bundle, tmp_path)
    loaded = load_bundle(tmp_path)

    assert loaded.sensors == bundle.sensors
    assert (loaded.window_length, loaded.rul_cap, loaded.alarm_threshold) == (10, 30, 0.42)
    assert np.array_equal(loaded.forest.score(windows[:5]), bundle.forest.score(windows[:5]))
    assert np.array_equal(loaded.normaliser.means_, bundle.normaliser.means_)


def test_saving_requires_the_model_to_be_in_the_folder(tmp_path):
    bundle, _ = make_bundle(tmp_path / "somewhere_else")

    with pytest.raises(ValueError, match="before saving"):
        save_bundle(bundle, tmp_path / "bundle")


def test_loading_without_a_model_explains_what_to_run(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    save_bundle(bundle, tmp_path)
    (tmp_path / ONNX_FILE).unlink()

    with pytest.raises(FileNotFoundError, match="prepare_stream_models"):
        load_bundle(tmp_path)


def test_bundle_files_can_be_loaded_with_joblib_directly(tmp_path):
    bundle, _ = make_bundle(tmp_path)
    save_bundle(bundle, tmp_path)

    assert joblib.load(tmp_path / "normaliser.joblib").n_conditions == 1
