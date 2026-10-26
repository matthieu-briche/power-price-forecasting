import numpy as np
import pandas as pd

from powerfc.trading import best_schedule, evaluate_trading, realised_pnl


def test_best_schedule_buys_low_sells_high_in_order():
    curve = np.full(24, 50.0)
    curve[3:5] = 10.0  # cheap night
    curve[18:20] = 150.0  # evening peak
    c, d, pnl = best_schedule(curve, hours=2, efficiency=1.0)
    assert (c, d) == (3, 18)
    assert pnl == 2 * 150 - 2 * 10


def test_no_trade_when_flat():
    c, d, pnl = best_schedule(np.full(24, 40.0), hours=2, efficiency=0.9)
    assert (c, d, pnl) == (-1, -1, 0.0)
    assert realised_pnl(np.full(24, 40.0), (c, d, pnl), 2, 0.9) == 0.0


def test_perfect_forecast_captures_everything():
    rng = np.random.default_rng(0)
    y = rng.normal(80, 30, 24 * 5)
    preds = pd.DataFrame({"date": np.repeat(np.arange(5), 24), "y": y, "perfect": y})
    res = evaluate_trading(preds, "perfect")
    assert res["capture_ratio"] == 1.0
