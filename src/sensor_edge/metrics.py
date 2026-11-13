"""RUL error metrics."""

import numpy as np


def rmse(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - true) ** 2)))


def nasa_score(true: np.ndarray, predicted: np.ndarray) -> float:
    """Asymmetric score from the PHM08 challenge. Lower is better.

    A late prediction (predicted RUL above the true RUL) means a failure the
    operator did not see coming, so it is penalised more than an early one.
    """
    error = predicted - true
    penalty = np.where(error < 0, np.exp(-error / 13) - 1, np.exp(error / 10) - 1)
    return float(penalty.sum())
