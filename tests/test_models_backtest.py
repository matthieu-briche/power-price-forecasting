import numpy as np
import pytest

from powerfc.backtest import BacktestConfig, run_backtest, smape
from powerfc.data import generate_synthetic
from powerfc.features import make_dataset
from powerfc.models import NaiveDayAhead, make_model


@pytest.fixture(scope="module")
def df():
    return generate_synthetic(start="2023-01-01", days=200, seed=7)


def test_naive_uses_weekly_lag_on_mondays_and_weekends(df):
    X, _ = make_dataset(df)
    p = NaiveDayAhead().predict(X)
    weekly = X["dow"].isin([0, 5, 6]).to_numpy()
    np.testing.assert_array_equal(p[weekly], X["price_lag_7d"].to_numpy()[weekly])
    np.testing.assert_array_equal(p[~weekly], X["price_lag_1d"].to_numpy()[~weekly])


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("does-not-exist")


def test_smape_handles_zero_prices():
    assert smape(np.array([0.0, 10.0]), np.array([0.0, 10.0])) == 0.0


def test_backtest_runs_and_ml_beats_naive(df):
    cfg = BacktestConfig(models=("naive", "ridge", "hgb"), test_days=21, train_window_days=120)
    preds, metrics = run_backtest(df, cfg)

    assert set(metrics.index) == {"naive", "ridge", "hgb"}
    assert preds["date"].nunique() == 21
    assert not preds.isna().any().any()
    assert metrics.loc["hgb", "mae"] < metrics.loc["naive", "mae"]
    assert metrics.loc["ridge", "rmae_vs_naive"] < 1.0
    assert (metrics["capture_ratio"] <= 1.0 + 1e-9).all()


def test_backtest_never_trains_on_test_day(df, monkeypatch):
    """Every fit call only sees data strictly before the day being forecast."""
    seen = []
    original = NaiveDayAhead.fit

    def spy(self, X, y=None):
        seen.append(X.index.max())
        return original(self, X, y)

    monkeypatch.setattr(NaiveDayAhead, "fit", spy)
    cfg = BacktestConfig(models=("naive",), test_days=14, retrain_every=1, train_window_days=60)
    preds, _ = run_backtest(df, cfg)
    first_test_hour = preds.index.min()
    assert all(ts < first_test_hour + np.timedelta64(i, "D") for i, ts in enumerate(seen))
