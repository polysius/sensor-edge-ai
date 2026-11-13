"""Classical RUL regressors that run on window features."""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_baselines(seed: int = 0) -> dict:
    return {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Gradient boosting": HistGradientBoostingRegressor(random_state=seed),
    }
