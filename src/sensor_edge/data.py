"""Load C-MAPSS files and split engine units."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit", "cycle", *SETTING_COLUMNS, *SENSOR_COLUMNS]

# FD001 and FD003 run at one operating condition. FD002 and FD004 mix six.
OPERATING_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}


@dataclass(frozen=True)
class Subset:
    """One C-MAPSS subset: run-to-failure training units and truncated test units."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # true RUL after the last test cycle, indexed by unit


def read_cmapss_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)


def load_subset(data_dir: Path, name: str) -> Subset:
    if name not in OPERATING_CONDITIONS:
        raise ValueError(f"Unknown subset {name!r}; expected one of {sorted(OPERATING_CONDITIONS)}")
    data_dir = Path(data_dir)
    train = read_cmapss_table(data_dir / f"train_{name}.txt")
    test = read_cmapss_table(data_dir / f"test_{name}.txt")
    rul_values = pd.read_csv(data_dir / f"RUL_{name}.txt", header=None)[0]
    # The RUL file lists one value per test unit, in unit order starting at 1.
    test_rul = pd.Series(rul_values.to_numpy(), index=sorted(test["unit"].unique()), name="rul")
    return Subset(name=name, train=train, test=test, test_rul=test_rul)


def split_units(units: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split engine units into train and validation sets.

    Splitting by unit keeps every cycle of an engine on one side, so no engine
    leaks from training into validation.
    """
    units = np.unique(units)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, round(len(units) * val_fraction))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])
