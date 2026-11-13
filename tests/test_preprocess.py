import numpy as np

from sensor_edge.data import SENSOR_COLUMNS
from sensor_edge.preprocess import (
    OperatingConditionNormaliser,
    add_capped_rul,
    select_informative_sensors,
)
from synthetic import CONSTANT_SENSOR, make_fleet


def test_normaliser_uses_only_training_statistics():
    train = make_fleet(n_units=10, seed=0)
    shifted = make_fleet(n_units=5, seed=1)
    shifted[SENSOR_COLUMNS] = shifted[SENSOR_COLUMNS] + 1000.0

    normaliser = OperatingConditionNormaliser(n_conditions=1).fit(train)
    means_after_fit = normaliser.means_.copy()
    transformed = normaliser.transform(shifted)

    assert np.array_equal(normaliser.means_, means_after_fit)
    # Shifted data stays far from zero because it is not re-centred on its own mean.
    assert transformed["sensor_2"].mean() > 100


def test_normaliser_centres_every_condition_separately():
    train = make_fleet(n_units=20, n_conditions=3, seed=0)

    normalised = OperatingConditionNormaliser(n_conditions=3).fit(train).transform(train)

    # Sensor 5 shifts with condition, so a per-condition fit leaves a mean of zero overall.
    assert abs(normalised["sensor_5"].mean()) < 1e-6
    assert abs(normalised["sensor_5"].std() - 1) < 0.2


def test_global_normalisation_leaves_condition_shift_in_the_data():
    train = make_fleet(n_units=20, n_conditions=3, seed=0)

    per_condition = OperatingConditionNormaliser(n_conditions=3).fit(train).transform(train)
    global_fit = OperatingConditionNormaliser(n_conditions=1).fit(train).transform(train)

    def largest_condition_mean(normalised):
        condition = (train["setting_1"] / 10).round()  # the fleet's conditions sit 10 units apart
        return normalised["sensor_5"].groupby(condition).mean().abs().max()

    assert largest_condition_mean(per_condition) < 1e-6
    assert largest_condition_mean(global_fit) > 0.5


def test_constant_sensor_is_dropped_from_the_data_not_a_fixed_list():
    train = make_fleet(n_units=10, seed=0)

    normalised = OperatingConditionNormaliser(n_conditions=1).fit(train).transform(train)
    sensors = select_informative_sensors(normalised)

    assert CONSTANT_SENSOR not in sensors
    assert len(sensors) == len(SENSOR_COLUMNS) - 1


def test_rul_counts_down_to_zero_and_is_capped():
    frame = make_fleet(n_units=3, seed=0)

    labelled = add_capped_rul(frame, cap=20)

    for _, engine in labelled.groupby("unit"):
        assert engine["rul"].iloc[-1] == 0
        assert engine["rul"].max() == 20
        assert engine["rul"].iloc[0] == 20  # engines here last longer than the cap
        assert engine["rul"].is_monotonic_decreasing
