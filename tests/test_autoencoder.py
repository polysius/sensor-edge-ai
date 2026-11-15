import numpy as np
import pytest

pytest.importorskip("torch")

from sensor_edge.autoencoder import AutoencoderDetector


def healthy_windows(n: int, seed: int) -> np.ndarray:
    """Windows with a fixed pattern across sensors, which a small autoencoder can learn."""
    rng = np.random.default_rng(seed)
    pattern = np.linspace(-1, 1, 4)
    level = rng.normal(size=(n, 1, 1))
    return (level * pattern[None, None, :] + rng.normal(0, 0.05, size=(n, 6, 4))).astype(np.float32)


def test_outliers_have_a_higher_reconstruction_error_than_healthy_windows():
    detector = AutoencoderDetector(seed=0, epochs=40).fit(healthy_windows(500, seed=0))
    healthy = healthy_windows(100, seed=1)
    broken = healthy.copy()
    broken[:, :, 0] += 4.0  # one sensor leaves the pattern the autoencoder learned

    assert detector.score(broken).mean() > 3 * detector.score(healthy).mean()


def test_score_returns_one_value_per_window():
    detector = AutoencoderDetector(seed=0, epochs=1).fit(healthy_windows(50, seed=0))

    assert detector.score(healthy_windows(7, seed=1)).shape == (7,)


def test_same_seed_gives_the_same_scores():
    train, probe = healthy_windows(100, seed=0), healthy_windows(10, seed=1)

    first = AutoencoderDetector(seed=3, epochs=2).fit(train).score(probe)
    second = AutoencoderDetector(seed=3, epochs=2).fit(train).score(probe)

    assert np.array_equal(first, second)


def test_scoring_before_fitting_is_an_error():
    with pytest.raises(RuntimeError, match="fit"):
        AutoencoderDetector().score(healthy_windows(3, seed=0))
