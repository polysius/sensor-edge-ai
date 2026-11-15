"""Anomaly detectors that learn what healthy windows look like.

Both detectors share one interface. `fit` sees healthy windows only, and `score`
returns one number per window where a higher value means more anomalous.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from sensor_edge.features import summarise_windows


class IsolationForestDetector:
    """Isolation forest on the same mean, std, slope and last-value features as the baselines."""

    def __init__(self, seed: int = 0, n_estimators: int = 200):
        self._forest = IsolationForest(n_estimators=n_estimators, random_state=seed)

    def fit(self, healthy_windows: np.ndarray) -> "IsolationForestDetector":
        self._forest.fit(summarise_windows(healthy_windows))
        return self

    def score(self, windows: np.ndarray) -> np.ndarray:
        # score_samples is higher for normal points, so flip the sign.
        return -self._forest.score_samples(summarise_windows(windows))
