import numpy as np
import pytest

from sensor_edge.anomaly_experiment import (
    END_OF_LIFE_RUL,
    _add_test_rul,
    _score_alarms,
    prepare_anomaly_data,
    run_anomaly_detection,
)
from sensor_edge.data import load_subset
from sensor_edge.detectors import IsolationForestDetector
from sensor_edge.experiment import ExperimentConfig
from sensor_edge.faults import inject_fault
from synthetic import write_subset

CONFIG = ExperimentConfig(window_length=10, rul_cap=30)


@pytest.fixture
def subset(tmp_path):
    write_subset(tmp_path, "FD001")
    return load_subset(tmp_path, "FD001")


def test_isolation_forest_scores_faulted_windows_higher():
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=(400, 10, 5))
    detector = IsolationForestDetector(seed=0).fit(healthy[:300])
    faulted = inject_fault(healthy[300:], "spike", np.random.default_rng(1))

    assert detector.score(faulted).mean() > detector.score(healthy[300:]).mean()


def test_test_rul_matches_the_official_value_at_the_last_cycle_and_counts_down(subset):
    labelled = _add_test_rul(subset, subset.test, rul_cap=10_000)

    for unit, engine in labelled.groupby("unit"):
        assert engine["rul"].iloc[-1] == subset.test_rul.loc[unit]
        assert (np.diff(engine["rul"].to_numpy()) == -1).all()


def test_test_rul_is_capped(subset):
    labelled = _add_test_rul(subset, subset.test, rul_cap=30)

    assert labelled["rul"].max() == 30


def test_anomaly_data_splits_healthy_and_degraded_windows(subset):
    data = prepare_anomaly_data(subset, CONFIG)

    for windows in (data.healthy_train, data.healthy_val, data.test_healthy, data.test_degraded):
        assert len(windows) > 0
        assert windows.shape[1] == CONFIG.window_length
    assert END_OF_LIFE_RUL <= CONFIG.rul_cap


def test_alarm_scores_on_a_hand_checked_example():
    healthy_scores = np.array([0.1, 0.2, 0.3, 0.9])  # the last one is a false alarm
    anomalous_scores = np.array([0.8, 0.95, 0.4])  # the last one is missed

    result = _score_alarms(healthy_scores, anomalous_scores, threshold=0.5)

    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["false_alarm_rate"] == pytest.approx(1 / 4)
    assert result["n_healthy"] == 4 and result["n_anomalous"] == 3


class LargestReadingDetector:
    """A trivial detector, so the evaluation can be checked without model quality in the way."""

    def fit(self, healthy_windows):
        return self

    def score(self, windows):
        return np.abs(windows).max(axis=(1, 2))


def test_run_anomaly_detection_reports_every_check_for_each_detector(subset):
    detectors = {"Largest reading": LargestReadingDetector}

    rows = run_anomaly_detection(subset, CONFIG, detectors)

    assert [row["check"] for row in rows] == ["spike", "drift", "stuck", "end of life"]
    spike = rows[0]
    assert spike["auc"] > 0.95  # a 5 std spike is the largest reading in its window
    assert spike["n_healthy"] == spike["n_anomalous"]
    assert 0 <= spike["false_alarm_rate"] <= 1
    assert rows[3]["n_anomalous"] != rows[3]["n_healthy"]  # end of life is not balanced


def test_isolation_forest_runs_through_the_evaluation_and_beats_chance_on_spikes(subset):
    detectors = {"Isolation forest": lambda: IsolationForestDetector(seed=0)}

    rows = run_anomaly_detection(subset, CONFIG, detectors)

    assert rows[0]["auc"] > 0.5
