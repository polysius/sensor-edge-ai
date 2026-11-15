import dataclasses

import numpy as np
import pytest

from synthetic import write_subset

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from sensor_edge.data import read_cmapss_table
from sensor_edge.experiment import ExperimentConfig
from sensor_edge.stream_models import load_bundle
from sensor_edge.stream_prepare import prepare_stream_bundle
from sensor_edge.stream_producer import readings_from_frame
from sensor_edge.stream_scoring import EngineScorer
from sensor_edge.train import TrainConfig

CONFIG = ExperimentConfig(window_length=10, rul_cap=30)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A small real bundle (trained on a synthetic fleet) and some readings to stream."""
    data_dir = tmp_path_factory.mktemp("data")
    write_subset(data_dir, "FD001")
    out_dir = tmp_path_factory.mktemp("bundle")
    prepare_stream_bundle(data_dir, "FD001", CONFIG, out_dir, TrainConfig(epochs=2, batch_size=64))
    frame = read_cmapss_table(data_dir / "test_FD001.txt")
    return out_dir, list(readings_from_frame(frame, faults=[]))


def test_prepare_saves_a_bundle_that_loads(prepared):
    out_dir, _ = prepared

    bundle = load_bundle(out_dir)

    assert bundle.window_length == CONFIG.window_length
    assert bundle.rul_cap == CONFIG.rul_cap
    assert np.isfinite(bundle.alarm_threshold)
    assert "sensor_1" not in bundle.sensors  # the constant sensor was dropped from the data


def test_scoring_a_reading_gives_finite_values_in_range(prepared):
    out_dir, readings = prepared
    scorer = EngineScorer(load_bundle(out_dir))

    result = scorer.score(readings[0])

    assert (result.unit, result.cycle) == (readings[0]["unit"], readings[0]["cycle"])
    assert 0 <= result.predicted_rul <= CONFIG.rul_cap
    assert np.isfinite(result.anomaly_score)


def test_an_engine_is_scored_the_same_whether_or_not_other_engines_share_the_stream(prepared):
    out_dir, readings = prepared
    engine_one = [r for r in readings if r["unit"] == 1][:15]

    alone = EngineScorer(load_bundle(out_dir))
    alone_results = [alone.score(r) for r in engine_one]
    mixed = EngineScorer(load_bundle(out_dir))
    mixed_results = [res for r in readings[:60] if (res := mixed.score(r)).unit == 1]

    assert [r.predicted_rul for r in mixed_results] == [r.predicted_rul for r in alone_results[: len(mixed_results)]]
    assert [r.anomaly_score for r in mixed_results] == [r.anomaly_score for r in alone_results[: len(mixed_results)]]


def test_scoring_is_deterministic(prepared):
    out_dir, readings = prepared

    first = [EngineScorer(load_bundle(out_dir)).score(r) for r in readings[:3]]
    second = [EngineScorer(load_bundle(out_dir)).score(r) for r in readings[:3]]

    assert first == second


@pytest.mark.parametrize("threshold, expected", [(-np.inf, True), (np.inf, False)])
def test_alert_follows_the_alarm_threshold(prepared, threshold, expected):
    out_dir, readings = prepared
    bundle = dataclasses.replace(load_bundle(out_dir), alarm_threshold=threshold)

    result = EngineScorer(bundle).score(readings[0])

    assert result.is_alert is expected
