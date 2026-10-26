"""Walk-forward back-testing that mimics daily day-ahead operations.

For every delivery day D in the test period:
  1. the model is (re)trained on a rolling window that ends on day D-1, every
     ``retrain_every`` days;
  2. it forecasts the 24 hourly prices of day D using only features available at
     gate closure (see ``features.py``).

No observation from day D or later is ever used to fit the model that forecasts day D.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from powerfc.features import LOCAL_TZ, make_dataset
from powerfc.models import make_model
from powerfc.trading import evaluate_trading

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    models: tuple[str, ...] = ("naive", "ridge", "hgb")
    test_days: int = 90
    train_window_days: int = 365
    retrain_every: int = 7
    battery_hours: int = 2
    battery_efficiency: float = 0.9


def mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((y - p) ** 2)))


def smape(y, p) -> float:
    """Symmetric MAPE in %, robust to the zero/negative prices that break plain MAPE."""
    denom = (np.abs(y) + np.abs(p)) / 2
    mask = denom > 1e-6
    return float(100 * np.mean(np.abs(y - p)[mask] / denom[mask]))


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (predictions, metrics). Predictions: one row per delivery hour."""
    X, y = make_dataset(df)
    date = pd.Series(X.index.tz_convert(LOCAL_TZ).normalize().tz_localize(None), index=X.index)
    days = np.sort(date.unique())
    if len(days) < cfg.test_days + 30:
        raise ValueError(f"Need more history: {len(days)} days for {cfg.test_days} test days")
    test_days = days[-cfg.test_days :]
    log.info("Back-test %s -> %s, models=%s", test_days[0], test_days[-1], cfg.models)

    out = []
    fitted = {}
    for i, d in enumerate(test_days):
        test_mask = (date == d).to_numpy()
        if i % cfg.retrain_every == 0:
            train_start = d - np.timedelta64(cfg.train_window_days, "D")
            train_mask = ((date < d) & (date >= train_start)).to_numpy()
            for name in cfg.models:
                fitted[name] = make_model(name).fit(X[train_mask], y[train_mask])
        day = pd.DataFrame({"date": d, "y": y[test_mask]}, index=X.index[test_mask])
        for name, model in fitted.items():
            day[name] = model.predict(X[test_mask])
        out.append(day)

    preds = pd.concat(out)
    preds.index.name = "timestamp"
    metrics = evaluate(preds, cfg)
    return preds, metrics


def evaluate(preds: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    rows = []
    naive_mae = mae(preds["y"], preds["naive"]) if "naive" in preds else np.nan
    for name in cfg.models:
        y, p = preds["y"].to_numpy(), preds[name].to_numpy()
        trade = evaluate_trading(preds, name, cfg.battery_hours, cfg.battery_efficiency)
        rows.append(
            {
                "model": name,
                "mae": mae(y, p),
                "rmse": rmse(y, p),
                "smape_pct": smape(y, p),
                "rmae_vs_naive": mae(y, p) / naive_mae if naive_mae else np.nan,
                **trade,
            }
        )
    return pd.DataFrame(rows).set_index("model").round(4)


def config_dict(cfg: BacktestConfig) -> dict:
    d = asdict(cfg)
    d["models"] = list(cfg.models)
    return d
