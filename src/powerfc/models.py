"""Forecasting models behind a common scikit-learn interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NaiveDayAhead(BaseEstimator, RegressorMixin):
    """Standard EPF naive benchmark (Lago et al., 2021).

    Tue-Fri: price of the same hour on D-1. Mon/Sat/Sun: same hour on D-7.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        weekly = X["dow"].isin([0, 5, 6]).to_numpy()
        return np.where(weekly, X["price_lag_7d"], X["price_lag_1d"]).astype(float)


def _ridge() -> Pipeline:
    cat = ["hour", "dow"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder=StandardScaler(),
    )
    return Pipeline([("pre", pre), ("model", RidgeCV(alphas=np.logspace(-2, 3, 12)))])


def _hgb(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",  # robust to price spikes; optimises the reported MAE
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


MODELS = {
    "naive": NaiveDayAhead,
    "ridge": _ridge,
    "hgb": _hgb,
}


def make_model(name: str):
    try:
        return MODELS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}") from exc
