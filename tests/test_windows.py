import numpy as np
import pandas as pd

from sensor_edge.windows import make_last_windows, make_training_windows


def engine_frame(unit: int, n_cycles: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit": unit,
            "cycle": np.arange(1, n_cycles + 1),
            "sensor_2": np.arange(1, n_cycles + 1, dtype=float),
            "rul": np.arange(n_cycles, 0, -1),
        }
    )


def test_training_windows_end_at_every_cycle():
    frame = pd.concat([engine_frame(1, 10), engine_frame(2, 6)])

    windows, targets, units = make_training_windows(frame, ["sensor_2"], window_length=4)

    assert windows.shape == (16, 4, 1)
    assert len(targets) == len(units) == 16
    # The window for cycle 8 of unit 1 holds cycles 5 to 8.
    assert windows[7, :, 0].tolist() == [5, 6, 7, 8]
    assert targets[7] == 3


def test_early_cycles_are_padded_with_the_first_reading():
    windows, _, _ = make_training_windows(engine_frame(1, 10), ["sensor_2"], window_length=4)

    assert windows[0, :, 0].tolist() == [1, 1, 1, 1]
    assert windows[1, :, 0].tolist() == [1, 1, 1, 2]


def test_last_window_scores_an_engine_shorter_than_the_window():
    frame = pd.concat([engine_frame(1, 3), engine_frame(2, 50)])

    windows, units = make_last_windows(frame, ["sensor_2"], window_length=30)

    assert windows.shape == (2, 30, 1)
    assert units.tolist() == [1, 2]
    assert windows[0, -3:, 0].tolist() == [1, 2, 3]
    assert (windows[0, :-3, 0] == 1).all()
    assert windows[1, -1, 0] == 50


def test_windows_do_not_mix_engines():
    frame = pd.concat([engine_frame(1, 5), engine_frame(2, 5)])
    frame.loc[frame["unit"] == 2, "sensor_2"] += 100

    windows, _, _ = make_training_windows(frame, ["sensor_2"], window_length=4)

    assert windows[:5].max() < 100  # no window of unit 1 reaches into unit 2
    assert windows[5:].min() > 100
