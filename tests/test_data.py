import numpy as np
import pytest

from sensor_edge.data import COLUMN_NAMES, load_subset, split_units
from synthetic import write_subset


def test_load_subset_reads_all_three_files(tmp_path):
    write_subset(tmp_path, "FD001")

    subset = load_subset(tmp_path, "FD001")

    assert list(subset.train.columns) == COLUMN_NAMES
    assert list(subset.test.columns) == COLUMN_NAMES
    assert list(subset.test_rul.index) == sorted(subset.test["unit"].unique())
    assert subset.test_rul.iloc[0] == 10


def test_load_subset_rejects_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="Unknown subset"):
        load_subset(tmp_path, "FD009")


def test_split_units_never_shares_a_unit():
    units = np.repeat(np.arange(1, 101), 5)  # 100 engines, 5 rows each

    train_units, val_units = split_units(units, val_fraction=0.2, seed=0)

    assert set(train_units).isdisjoint(val_units)
    assert set(train_units) | set(val_units) == set(range(1, 101))
    assert len(val_units) == 20


def test_split_units_is_reproducible_and_seed_dependent():
    units = np.arange(1, 101)

    first = split_units(units, 0.2, seed=0)
    again = split_units(units, 0.2, seed=0)
    other = split_units(units, 0.2, seed=1)

    assert np.array_equal(first[1], again[1])
    assert not np.array_equal(first[1], other[1])
