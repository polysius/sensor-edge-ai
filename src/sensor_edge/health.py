"""A one-number health indicator built from the normalised sensors."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SMOOTHING_WIDTH = 5  # cycles averaged to damp sensor noise


def _trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    return pd.Series(values).rolling(width, min_periods=1).mean().to_numpy()


class HealthIndicator:
    """First principal component of the sensors, oriented so it rises as an engine wears.

    The component and the failure level are fitted on training engines only.
    Each engine's trajectory is smoothed with a trailing mean, so a value never
    uses readings from later cycles.
    """

    def __init__(self, sensors: list[str], rul_cap: int):
        self.sensors = sensors
        self.rul_cap = rul_cap
        self._pca = PCA(n_components=1)
        self._sign = 1.0
        self.failure_level_: float | None = None

    def fit(self, train: pd.DataFrame) -> "HealthIndicator":
        """`train` is a normalised run-to-failure table with a capped `rul` column."""
        self._pca.fit(train[self.sensors].to_numpy())
        raw = self._raw(train)
        near_failure = raw[(train["rul"] == 0).to_numpy()].mean()
        healthy = raw[(train["rul"] >= self.rul_cap).to_numpy()].mean()
        self._sign = 1.0 if near_failure >= healthy else -1.0

        final_levels = [
            self.trajectory(engine)[1][-1] for _, engine in train.groupby("unit")
        ]
        self.failure_level_ = float(np.mean(final_levels))
        return self

    def trajectory(self, engine: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Cycle numbers and smoothed indicator values for one engine."""
        engine = engine.sort_values("cycle")
        values = _trailing_mean(self._raw(engine), SMOOTHING_WIDTH)
        return engine["cycle"].to_numpy(), values

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self._sign * self._pca.transform(frame[self.sensors].to_numpy())[:, 0]
