"""Inject synthetic faults into healthy windows.

C-MAPSS has no fault labels, so detectors are scored on windows where a known
fault was added. Windows hold normalised readings, so one unit is one standard
deviation of that sensor under its operating condition.
"""

import numpy as np

FAULT_TYPES = ["spike", "drift", "stuck"]

SPIKE_SIZE = 5.0  # standard deviations
SPIKE_CYCLES = 3  # the spike lasts for the last few cycles of the window
DRIFT_SIZE = 4.0  # standard deviations reached at the end of the window


def inject_fault(windows: np.ndarray, fault_type: str, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `windows` with one fault of `fault_type` in one random sensor each.

    spike: the sensor jumps by `SPIKE_SIZE` for the last few cycles.
    drift: the sensor ramps from 0 up to `DRIFT_SIZE` across the window.
    stuck: the sensor freezes at its first reading in the window.
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unknown fault type {fault_type!r}; expected one of {FAULT_TYPES}")
    faulted = windows.copy()
    n_windows, window_length, n_sensors = windows.shape
    sensor = rng.integers(0, n_sensors, size=n_windows)
    rows = np.arange(n_windows)

    if fault_type == "spike":
        direction = rng.choice([-1.0, 1.0], size=n_windows)
        faulted[rows, -SPIKE_CYCLES:, sensor] += direction[:, None] * SPIKE_SIZE
    elif fault_type == "drift":
        direction = rng.choice([-1.0, 1.0], size=n_windows)
        ramp = np.linspace(0.0, DRIFT_SIZE, window_length)
        faulted[rows, :, sensor] += direction[:, None] * ramp[None, :]
    else:
        first_reading = windows[rows, 0, sensor]
        faulted[rows, :, sensor] = first_reading[:, None]
    return faulted
