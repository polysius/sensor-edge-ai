"""Hand-crafted window features for the classical baselines."""

import numpy as np


def summarise_windows(windows: np.ndarray) -> np.ndarray:
    """Reduce each (window, sensors) block to mean, std, slope and last value per sensor.

    Returns an array of shape (n_windows, 4 * n_sensors). The slope is the least
    squares trend per cycle, which captures how fast a sensor is drifting.
    """
    window_length = windows.shape[1]
    time = np.arange(window_length) - (window_length - 1) / 2
    centred = windows - windows.mean(axis=1, keepdims=True)
    slope = np.einsum("t,nts->ns", time, centred) / np.sum(time**2)
    return np.concatenate(
        [windows.mean(axis=1), windows.std(axis=1), slope, windows[:, -1, :]], axis=1
    )
