"""Score anomaly detectors on injected faults and on engines close to failure."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from sensor_edge.data import Subset
from sensor_edge.experiment import ExperimentConfig, normalise_subset
from sensor_edge.faults import FAULT_TYPES, inject_fault
from sensor_edge.windows import make_training_windows

# A window with at most this much RUL left counts as degraded in the end-of-life check.
END_OF_LIFE_RUL = 30
# The alarm threshold is the score that this share of healthy validation windows stays under.
THRESHOLD_QUANTILE = 0.95


class Detector(Protocol):
    def fit(self, healthy_windows: np.ndarray): ...

    def score(self, windows: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AnomalyData:
    """Windows used to train, set the threshold for, and evaluate detectors.

    A window is healthy while its capped RUL still equals the cap, which means
    degradation has not started to show.
    """

    healthy_train: np.ndarray  # training engines, fits the detector
    healthy_val: np.ndarray  # validation engines, sets the alarm threshold
    test_healthy: np.ndarray  # test engines, negatives for every evaluation
    test_degraded: np.ndarray  # test engines within END_OF_LIFE_RUL of failure


def _add_test_rul(subset: Subset, test_frame, rul_cap: int):
    """Label every test cycle with its RUL.

    The RUL file gives the RUL after the last observed cycle, so earlier cycles
    have that many cycles plus the number of observed cycles still to come.
    """
    last_cycle = test_frame.groupby("unit")["cycle"].transform("max")
    final_rul = test_frame["unit"].map(subset.test_rul)
    labelled = test_frame.copy()
    labelled["rul"] = (final_rul + last_cycle - test_frame["cycle"]).clip(upper=rul_cap)
    return labelled


def prepare_anomaly_data(subset: Subset, config: ExperimentConfig) -> AnomalyData:
    frames = normalise_subset(subset, config)
    test = _add_test_rul(subset, frames.test, config.rul_cap)

    def windows_with_rul(frame):
        windows, rul, _ = make_training_windows(frame, frames.sensors, config.window_length)
        return windows, rul

    train_windows, train_rul = windows_with_rul(frames.train)
    val_windows, val_rul = windows_with_rul(frames.val)
    test_windows, test_rul = windows_with_rul(test)
    return AnomalyData(
        healthy_train=train_windows[train_rul >= config.rul_cap],
        healthy_val=val_windows[val_rul >= config.rul_cap],
        test_healthy=test_windows[test_rul >= config.rul_cap],
        test_degraded=test_windows[test_rul <= END_OF_LIFE_RUL],
    )


def _score_alarms(
    healthy_scores: np.ndarray, anomalous_scores: np.ndarray, threshold: float
) -> dict:
    """Precision, recall and F1 for the alarm rule `score > threshold`, plus threshold-free AUC."""
    labels = np.concatenate([np.zeros(len(healthy_scores)), np.ones(len(anomalous_scores))])
    scores = np.concatenate([healthy_scores, anomalous_scores])
    alarms = scores > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, alarms, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc_score(labels, scores)),
        "false_alarm_rate": float((healthy_scores > threshold).mean()),
        "n_healthy": len(healthy_scores),
        "n_anomalous": len(anomalous_scores),
    }


def run_anomaly_detection(
    subset: Subset,
    config: ExperimentConfig,
    detector_factories: dict[str, Callable[[], Detector]],
    seed: int = 0,
) -> list[dict]:
    """Fit each detector on healthy training windows and score it on the test engines.

    Two checks are reported. Injected faults put a known fault into a copy of every
    healthy test window, one fault type at a time, so each row is balanced one to one.
    End of life asks whether windows within 30 cycles of failure raise alarms, and
    that check has more healthy windows than degraded ones.
    """
    data = prepare_anomaly_data(subset, config)
    rows = []
    for name, make_detector in detector_factories.items():
        detector = make_detector().fit(data.healthy_train)
        threshold = float(np.quantile(detector.score(data.healthy_val), THRESHOLD_QUANTILE))
        healthy_scores = detector.score(data.test_healthy)

        for fault_type in FAULT_TYPES:
            fault_rng = np.random.default_rng(seed)
            faulted = inject_fault(data.test_healthy, fault_type, fault_rng)
            result = _score_alarms(healthy_scores, detector.score(faulted), threshold)
            rows.append({"subset": subset.name, "detector": name, "check": fault_type, **result})

        end_of_life = _score_alarms(healthy_scores, detector.score(data.test_degraded), threshold)
        rows.append(
            {"subset": subset.name, "detector": name, "check": "end of life", **end_of_life}
        )
    return rows
