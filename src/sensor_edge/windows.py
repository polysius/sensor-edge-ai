"""Turn per-engine sensor histories into fixed-length windows."""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _windows_for_unit(readings: np.ndarray, window_length: int) -> np.ndarray:
    """Return one window ending at every cycle of a single engine.

    Cycles earlier than `window_length` have a short history. The first reading
    is repeated at the front so those cycles still get a full window, and the
    test engines that are shorter than the window can be scored too.
    """
    front_padding = np.repeat(readings[:1], window_length - 1, axis=0)
    padded = np.concatenate([front_padding, readings])
    windows = sliding_window_view(padded, window_length, axis=0)  # (cycles, sensors, window)
    return windows.transpose(0, 2, 1)


def make_training_windows(
    frame: pd.DataFrame, sensors: list[str], window_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows ending at every cycle, with the RUL target and the unit of each window."""
    ordered = frame.sort_values(["unit", "cycle"])
    windows, targets, units = [], [], []
    for unit, engine in ordered.groupby("unit"):
        engine_windows = _windows_for_unit(engine[sensors].to_numpy(), window_length)
        windows.append(engine_windows)
        targets.append(engine["rul"].to_numpy())
        units.append(np.full(len(engine), unit))
    return np.concatenate(windows), np.concatenate(targets), np.concatenate(units)


def make_last_windows(
    frame: pd.DataFrame, sensors: list[str], window_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """One window per engine, ending at its final cycle. This is what test scoring uses."""
    ordered = frame.sort_values(["unit", "cycle"])
    windows, units = [], []
    for unit, engine in ordered.groupby("unit"):
        windows.append(_windows_for_unit(engine[sensors].to_numpy(), window_length)[-1])
        units.append(unit)
    return np.stack(windows), np.array(units)
