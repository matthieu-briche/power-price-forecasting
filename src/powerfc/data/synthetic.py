"""Synthetic but structurally realistic hourly power-market data.

Used for tests, CI and offline demos. The generator reproduces the stylised facts that
matter for day-ahead price forecasting:

* daily and weekly load profiles, winter/summer seasonality;
* solar generation with a midday peak scaled by season and daily cloudiness;
* persistent (AR(1)) wind regimes;
* a convex merit-order curve driven by residual load (load - wind - solar) and a
  random-walk fuel price, producing negative prices at high renewable infeed and
  price spikes in scarcity hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Europe/Berlin"


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = np.empty(n)
    x[0] = rng.normal(0, sigma / np.sqrt(1 - phi**2))
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def generate_synthetic(start: str = "2023-01-01", days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Return an hourly DataFrame (UTC index) with columns price, load, wind, solar."""
    rng = np.random.default_rng(seed)
    idx_local = pd.date_range(start, periods=days * 24, freq="h", tz=LOCAL_TZ)
    n = len(idx_local)
    hour = idx_local.hour.to_numpy()
    dow = idx_local.dayofweek.to_numpy()
    doy = idx_local.dayofyear.to_numpy()
    day_id = (idx_local.normalize() - idx_local.normalize()[0]).days.to_numpy()

    season = np.cos(2 * np.pi * (doy - 15) / 365.25)  # +1 mid-winter, -1 mid-summer

    # Load (GW): morning/evening peaks, weekend dip, winter uplift.
    daily = 0.35 * np.exp(-((hour - 12) ** 2) / 30) + 0.12 * np.exp(-((hour - 19) ** 2) / 4)
    weekend = np.where(dow >= 5, -0.12, 0.0)
    load = 50 * (1 + daily + weekend + 0.08 * season) + _ar1(n, 0.95, 0.6, rng)

    # Solar (GW): daylight bell curve, longer/stronger days in summer, daily cloudiness.
    daylight = np.clip(np.sin(np.pi * (hour - 5) / 15), 0, None) ** 1.5
    cloud = np.clip(0.75 + _ar1(days, 0.6, 0.18, rng), 0.15, 1.0)[day_id]
    solar = 55 * daylight * (0.55 - 0.4 * season) * cloud

    # Wind (GW): persistent weather regimes, stronger in winter.
    wind_state = _ar1(n, 0.985, 0.18, rng)
    wind = 65 * (1 / (1 + np.exp(-(wind_state - 0.6 + 0.5 * season)))) * 0.85

    residual = load - wind - solar

    # Fuel price (EUR/MWh_th): daily random walk around 35.
    gas = 35 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))[day_id]

    # Convex merit order: cheap renewables/nuclear/lignite, then gas, then scarcity peakers.
    price = (
        2.0 * gas * np.clip(residual, 0, 60) / 40
        + 25 * np.exp((residual - 58) / 5)
        + np.where(residual < 5, 3 * (np.clip(residual, -25, None) - 5), 0)
        + _ar1(n, 0.8, 4.0, rng)
    )

    df = pd.DataFrame(
        {"price": price, "load": load, "wind": wind, "solar": solar}, index=idx_local
    ).round(3)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df
