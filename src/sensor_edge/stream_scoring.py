"""Score a stream of sensor readings one reading at a time, with no Kafka involved."""

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sensor_edge.data import SENSOR_COLUMNS, SETTING_COLUMNS
from sensor_edge.edge_bench import create_session, predict_scaled, scaled_to_rul
from sensor_edge.stream_models import StreamBundle

READING_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS


@dataclass(frozen=True)
class ScoreResult:
    unit: int
    cycle: int
    predicted_rul: float  # cycles of remaining life
    anomaly_score: float  # higher means less like healthy data
    is_alert: bool  # the anomaly score is above the alarm threshold


class EngineScorer:
    """Keeps the recent readings of every engine and scores each new reading.

    A reading is normalised with the statistics fitted on training engines, added to
    its engine's history, and the last `window_length` readings are scored. An engine
    with a short history is padded at the front with its first reading, as in training.
    """

    def __init__(self, bundle: StreamBundle):
        self._bundle = bundle
        self._session = create_session(bundle.onnx_path)
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=bundle.window_length))

    def score(self, reading: dict) -> ScoreResult:
        unit, cycle = int(reading["unit"]), int(reading["cycle"])
        normalised = self._normalise(reading)
        history = self._history[unit]
        history.append(normalised)

        window = self._window(history)[np.newaxis]
        predicted = scaled_to_rul(predict_scaled(self._session, window), self._bundle.rul_cap)[0]
        anomaly_score = float(self._bundle.forest.score(window)[0])
        return ScoreResult(
            unit=unit,
            cycle=cycle,
            predicted_rul=float(predicted),
            anomaly_score=anomaly_score,
            is_alert=anomaly_score > self._bundle.alarm_threshold,
        )

    def _normalise(self, reading: dict) -> np.ndarray:
        frame = pd.DataFrame([{column: reading[column] for column in READING_COLUMNS}])
        normalised = self._bundle.normaliser.transform(frame)
        return normalised[self._bundle.sensors].to_numpy(dtype=np.float32)[0]

    def _window(self, history: deque) -> np.ndarray:
        readings = list(history)
        missing = self._bundle.window_length - len(readings)
        return np.stack([readings[0]] * missing + readings)
