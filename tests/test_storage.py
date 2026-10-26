import mongomock
import pandas as pd
import pytest

from powerfc.backtest import BacktestConfig, config_dict, run_backtest
from powerfc.data import generate_synthetic
from powerfc.storage import MongoStore


@pytest.fixture
def store():
    return MongoStore(client=mongomock.MongoClient(), db_name="test")


def test_upsert_is_idempotent_and_roundtrips(store):
    df = generate_synthetic(start="2024-01-01", days=3, seed=0)
    store.upsert_market_data(df)
    store.upsert_market_data(df)  # second ingest must not duplicate
    assert store.db.market_data.count_documents({}) == len(df)

    back = store.load_market_data()
    pd.testing.assert_frame_equal(back, df, check_freq=False, check_names=False)


def test_load_date_filter(store):
    df = generate_synthetic(start="2024-01-01", days=3, seed=0)
    store.upsert_market_data(df)
    part = store.load_market_data(start="2024-01-02", end="2024-01-03")
    assert len(part) == 24


def test_save_run_and_report(store):
    df = generate_synthetic(start="2023-01-01", days=120, seed=0)
    cfg = BacktestConfig(models=("naive", "ridge"), test_days=5, train_window_days=60)
    preds, metrics = run_backtest(df, cfg)
    run_id = store.save_run(preds, metrics, config_dict(cfg), "DE-LU")

    assert store.db.runs.count_documents({"run_id": run_id}) == 1
    assert store.db.forecasts.count_documents({"run_id": run_id}) == 2 * len(preds)
