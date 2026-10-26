import numpy as np
import pandas as pd
import pytest

from powerfc.data import generate_synthetic
from powerfc.features import LOCAL_TZ, build_features, make_dataset


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(start="2024-01-01", days=60, seed=1)


def test_no_lookahead_leakage(df):
    """Changing prices on/after delivery day D must not change features for day D."""
    local_date = df.index.tz_convert(LOCAL_TZ).normalize()
    d = local_date.unique()[30]
    X_ref = build_features(df)

    perturbed = df.copy()
    perturbed.loc[local_date >= d, "price"] += 1000.0
    X_new = build_features(perturbed)

    rows = local_date == d
    pd.testing.assert_frame_equal(X_ref[rows], X_new[rows])


def test_lag_alignment(df):
    X = build_features(df)
    local = df.index.tz_convert(LOCAL_TZ)
    i = 24 * 20 + 13  # a mid-day hour well past warm-up, no DST in January/February
    assert local[i - 24].hour == local[i].hour
    assert X["price_lag_1d"].iloc[i] == df["price"].iloc[i - 24]
    assert X["price_lag_7d"].iloc[i] == df["price"].iloc[i - 168]


def test_dataset_drops_warmup(df):
    X, y = make_dataset(df)
    assert not X.isna().any().any()
    assert len(X) == len(y)
    assert len(X) <= len(df) - 7 * 24


def test_dst_days_handled():
    df = generate_synthetic(start="2024-03-20", days=20, seed=3)  # spans 31 Mar 2024 DST switch
    X = build_features(df)
    assert np.isfinite(X.dropna().to_numpy(dtype=float)).all()
