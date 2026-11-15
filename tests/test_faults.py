import numpy as np
import pytest

from sensor_edge.faults import DRIFT_SIZE, SPIKE_CYCLES, SPIKE_SIZE, inject_fault

N_WINDOWS, WINDOW, N_SENSORS = 50, 12, 6


@pytest.fixture
def healthy():
    return np.random.default_rng(0).normal(size=(N_WINDOWS, WINDOW, N_SENSORS))


def changed_sensors(original, faulted):
    """For each window, the sensors whose values differ."""
    return [np.flatnonzero((original[i] != faulted[i]).any(axis=0)) for i in range(len(original))]


def test_input_is_not_modified(healthy):
    before = healthy.copy()

    inject_fault(healthy, "spike", np.random.default_rng(1))

    assert np.array_equal(healthy, before)


@pytest.mark.parametrize("fault_type", ["spike", "drift"])
def test_additive_faults_touch_exactly_one_sensor(healthy, fault_type):
    faulted = inject_fault(healthy, fault_type, np.random.default_rng(1))

    assert all(len(sensors) == 1 for sensors in changed_sensors(healthy, faulted))


def test_spike_shifts_only_the_last_cycles_by_the_spike_size(healthy):
    faulted = inject_fault(healthy, "spike", np.random.default_rng(1))

    difference = faulted - healthy
    for i, (window_diff, sensors) in enumerate(zip(difference, changed_sensors(healthy, faulted))):
        column = window_diff[:, sensors[0]]
        assert np.allclose(np.abs(column[-SPIKE_CYCLES:]), SPIKE_SIZE)
        assert np.allclose(column[:-SPIKE_CYCLES], 0), i


def test_drift_ramps_from_zero_to_the_drift_size(healthy):
    faulted = inject_fault(healthy, "drift", np.random.default_rng(1))

    difference = faulted - healthy
    for window_diff, sensors in zip(difference, changed_sensors(healthy, faulted)):
        column = window_diff[:, sensors[0]]
        assert column[0] == pytest.approx(0)
        assert abs(column[-1]) == pytest.approx(DRIFT_SIZE)
        assert np.all(np.diff(np.abs(column)) > 0)


def test_stuck_sensor_is_frozen_at_its_first_reading(healthy):
    faulted = inject_fault(healthy, "stuck", np.random.default_rng(1))

    frozen_sensors = [
        np.flatnonzero(np.ptp(window, axis=0) == 0) for window in faulted
    ]
    assert all(len(sensors) == 1 for sensors in frozen_sensors)
    for original, window, sensors in zip(healthy, faulted, frozen_sensors):
        assert window[0, sensors[0]] == original[0, sensors[0]]


def test_same_seed_gives_the_same_faults(healthy):
    first = inject_fault(healthy, "drift", np.random.default_rng(3))
    second = inject_fault(healthy, "drift", np.random.default_rng(3))

    assert np.array_equal(first, second)


def test_unknown_fault_type_is_rejected(healthy):
    with pytest.raises(ValueError, match="Unknown fault type"):
        inject_fault(healthy, "melt", np.random.default_rng(0))
