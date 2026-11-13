"""Per-operating-condition normalisation, sensor selection and the RUL target."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sensor_edge.data import SENSOR_COLUMNS, SETTING_COLUMNS

# Below this standard deviation after normalisation a sensor carries no signal.
MIN_SENSOR_STD = 1e-6


class OperatingConditionNormaliser:
    """Standardise each sensor separately within every operating condition.

    Sensor readings shift with altitude, Mach number and throttle, so a global
    z-score would mix those shifts into the degradation signal. The conditions
    are found by clustering the three operating settings. Everything is fitted
    on training data only and then applied unchanged to validation and test data.
    """

    def __init__(self, n_conditions: int, seed: int = 0):
        self.n_conditions = n_conditions
        self.seed = seed
        self._clusterer: KMeans | None = None
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "OperatingConditionNormaliser":
        if self.n_conditions > 1:
            self._clusterer = KMeans(self.n_conditions, n_init=10, random_state=self.seed)
            self._clusterer.fit(train[SETTING_COLUMNS].to_numpy())
        conditions = self._assign_conditions(train)
        sensors = train[SENSOR_COLUMNS].to_numpy()
        self.means_ = np.zeros((self.n_conditions, len(SENSOR_COLUMNS)))
        self.stds_ = np.ones((self.n_conditions, len(SENSOR_COLUMNS)))
        for condition in range(self.n_conditions):
            rows = sensors[conditions == condition]
            if len(rows) == 0:
                continue
            self.means_[condition] = rows.mean(axis=0)
            std = rows.std(axis=0)
            # A constant sensor has zero spread. Dividing by 1 keeps it at exactly 0.
            self.stds_[condition] = np.where(std < 1e-12, 1.0, std)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("Call fit before transform")
        conditions = self._assign_conditions(frame)
        normalised = frame.copy()
        sensors = frame[SENSOR_COLUMNS].to_numpy()
        normalised[SENSOR_COLUMNS] = (sensors - self.means_[conditions]) / self.stds_[conditions]
        return normalised

    def _assign_conditions(self, frame: pd.DataFrame) -> np.ndarray:
        if self._clusterer is None:
            return np.zeros(len(frame), dtype=int)
        return self._clusterer.predict(frame[SETTING_COLUMNS].to_numpy())


def select_informative_sensors(normalised_train: pd.DataFrame) -> list[str]:
    """Keep sensors that still vary after normalisation.

    The choice is made from the data, because the set of constant sensors
    differs between subsets.
    """
    spread = normalised_train[SENSOR_COLUMNS].std()
    return [sensor for sensor in SENSOR_COLUMNS if spread[sensor] > MIN_SENSOR_STD]


def add_capped_rul(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Add a piecewise linear RUL target to run-to-failure data.

    Early in an engine's life degradation is not visible, so the target stays
    flat at `cap` and only starts to fall once the engine is near failure.
    """
    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    labelled = frame.copy()
    labelled["rul"] = (last_cycle - frame["cycle"]).clip(upper=cap)
    return labelled
