"""Small synthetic engine fleets in the C-MAPSS column layout."""

from pathlib import Path

import numpy as np
import pandas as pd

from sensor_edge.data import COLUMN_NAMES, SENSOR_COLUMNS

# Distinct operating settings, one row per condition.
CONDITION_SETTINGS = np.array(
    [[0.0, 0.0, 100.0], [10.0, 0.25, 100.0], [20.0, 0.70, 60.0], [35.0, 0.84, 100.0]]
)
CONSTANT_SENSOR = "sensor_1"


def make_fleet(n_units: int, n_conditions: int = 1, seed: int = 0) -> pd.DataFrame:
    """Run-to-failure engines whose sensors drift with age and shift with condition."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        n_cycles = int(rng.integers(40, 80))
        cycles = np.arange(1, n_cycles + 1)
        condition = rng.integers(0, n_conditions, size=n_cycles)
        frame = pd.DataFrame({"unit": unit, "cycle": cycles})
        settings = CONDITION_SETTINGS[condition] + rng.normal(0, 0.001, size=(n_cycles, 3))
        for i, name in enumerate(["setting_1", "setting_2", "setting_3"]):
            frame[name] = settings[:, i]
        for i, sensor in enumerate(SENSOR_COLUMNS):
            offset = 50.0 * condition * (i % 3)
            wear = 0.1 * cycles * (i % 4 - 1)
            frame[sensor] = offset + wear + rng.normal(0, 0.5, size=n_cycles) + i
        frame[CONSTANT_SENSOR] = 518.67
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[COLUMN_NAMES]


def write_subset(data_dir: Path, name: str, n_conditions: int = 1) -> None:
    """Write train, test and RUL files for a tiny subset in the real file format."""
    train = make_fleet(n_units=20, n_conditions=n_conditions, seed=1)
    test_full = make_fleet(n_units=8, n_conditions=n_conditions, seed=2)
    # Cut the last 15 cycles of every engine, as the real test files stop before failure.
    cycles_left = test_full.groupby("unit").cumcount(ascending=False)
    test = test_full[cycles_left >= 15]
    rul = np.arange(10, 10 + test["unit"].nunique())

    for path, table in ((data_dir / f"train_{name}.txt", train), (data_dir / f"test_{name}.txt", test)):
        table.to_csv(path, sep=" ", header=False, index=False)
    pd.Series(rul).to_csv(data_dir / f"RUL_{name}.txt", header=False, index=False)
