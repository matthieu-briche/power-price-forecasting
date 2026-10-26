"""Turn forecasts into trading decisions and measure their economic value.

Strategy: a 1 MW / ``hours`` MWh battery doing one cycle per day on the day-ahead
market. Using the *forecast* curve, pick the charge window and a later discharge window
that maximise the expected spread; settle the schedule at *realised* prices.

The capture ratio (realised PnL / perfect-foresight PnL) shows how much of the available
value a forecast actually captures: two forecasts with equal MAE can differ a lot here,
because what matters is ranking the cheap and expensive hours correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def best_schedule(
    curve: np.ndarray, hours: int = 2, efficiency: float = 0.9
) -> tuple[int, int, float]:
    """Return (charge_start, discharge_start, expected_pnl) for one daily curve.

    Charging buys ``hours`` MWh; discharging sells ``efficiency * hours`` MWh.
    Returns pnl 0 and (-1, -1) if no window is profitable.
    """
    window = np.convolve(curve, np.ones(hours), mode="valid")  # sum over each window
    best = (-1, -1, 0.0)
    for c in range(len(window)):
        for d in range(c + hours, len(window)):
            pnl = efficiency * window[d] - window[c]
            if pnl > best[2]:
                best = (c, d, float(pnl))
    return best


def realised_pnl(
    actual: np.ndarray, schedule: tuple[int, int, float], hours: int, efficiency: float
) -> float:
    c, d, _ = schedule
    if c < 0:
        return 0.0
    return float(efficiency * actual[d : d + hours].sum() - actual[c : c + hours].sum())


def evaluate_trading(
    preds: pd.DataFrame, model_col: str, hours: int = 2, efficiency: float = 0.9
) -> dict:
    """``preds`` has columns ['date', 'y', model_col] with 24 rows per delivery day."""
    pnl, perfect = 0.0, 0.0
    days = 0
    for _, day in preds.groupby("date"):
        if len(day) != 24:  # skip DST-transition days for simplicity
            continue
        actual = day["y"].to_numpy()
        sched = best_schedule(day[model_col].to_numpy(), hours, efficiency)
        pnl += realised_pnl(actual, sched, hours, efficiency)
        perfect += best_schedule(actual, hours, efficiency)[2]
        days += 1
    return {
        "days": days,
        "pnl_eur": round(pnl, 2),
        "perfect_foresight_pnl_eur": round(perfect, 2),
        "capture_ratio": round(pnl / perfect, 4) if perfect > 0 else float("nan"),
    }
