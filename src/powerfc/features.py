"""Feature engineering that respects the day-ahead information set.

The day-ahead auction for delivery day D closes at 12:00 CET on D-1. At that moment:

* all prices of day D-1 are known (they were set by the D-2 auction);
* prices of day D are, by definition, unknown;
* load / wind / solar for day D are only available as *forecasts*.

Every price-derived feature for a target hour on day D therefore uses only prices from
day D-1 or earlier. ``tests/test_features.py`` enforces this with a perturbation test:
changing any price on or after day D must leave the features of day D unchanged.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Europe/Berlin"
PRICE_LAGS_DAYS = (1, 2, 7)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix (same index as ``df``) for every delivery hour."""
    local = df.index.tz_convert(LOCAL_TZ)
    X = pd.DataFrame(index=df.index)

    # Calendar (local market time).
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)
    X["month"] = local.month

    # Fundamentals for the delivery hour (day-ahead forecasts in production).
    X["load"] = df["load"]
    X["wind"] = df["wind"]
    X["solar"] = df["solar"]
    X["residual_load"] = df["load"] - df["wind"] - df["solar"]
    X["renewable_share"] = (df["wind"] + df["solar"]) / df["load"]

    # Same-hour price lags: D-1, D-2, D-7, aligned on local date + hour (DST-safe).
    key = pd.MultiIndex.from_arrays([local.normalize().tz_localize(None), local.hour])
    price_by_key = pd.Series(df["price"].to_numpy(), index=key)
    price_by_key = price_by_key[~price_by_key.index.duplicated()]  # DST fall-back hour
    for d in PRICE_LAGS_DAYS:
        lag_key = pd.MultiIndex.from_arrays(
            [local.normalize().tz_localize(None) - pd.Timedelta(days=d), local.hour]
        )
        X[f"price_lag_{d}d"] = price_by_key.reindex(lag_key).to_numpy()

    # Daily statistics of D-1 prices (complete day, known at gate closure).
    date = local.normalize().tz_localize(None)
    daily = df["price"].groupby(date).agg(["mean", "min", "max", "std"])
    prev = daily.shift(1, freq="D").reindex(date)
    for col in ["mean", "min", "max", "std"]:
        X[f"price_d1_{col}"] = prev[col].to_numpy()

    # Residual-load change vs. same hour yesterday: drives day-on-day price moves.
    resid_by_key = pd.Series(X["residual_load"].to_numpy(), index=key)
    resid_by_key = resid_by_key[~resid_by_key.index.duplicated()]
    lag1_key = pd.MultiIndex.from_arrays([date - pd.Timedelta(days=1), local.hour])
    X["residual_load_delta_1d"] = (
        X["residual_load"].to_numpy() - resid_by_key.reindex(lag1_key).to_numpy()
    )

    return X


def make_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target with warm-up rows (missing lags) removed."""
    X = build_features(df)
    y = df["price"].rename("y")
    mask = X.notna().all(axis=1)
    return X[mask], y[mask]
