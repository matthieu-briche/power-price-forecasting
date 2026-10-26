"""Loader for public market data from the Energy-Charts API (Fraunhofer ISE).

Docs: https://api.energy-charts.info/  (free, no API key)

* ``/price?bzn=DE-LU``          day-ahead auction prices (EUR/MWh)
* ``/public_power?country=de``  realised load and generation by technology (MW)

Notes
-----
* Since 1 Oct 2025 the SDAC day-ahead auction runs on a 15-minute MTU; every series is
  resampled to hourly means so the rest of the pipeline works on one hourly grid.
* ``public_power`` returns *realised* wind/solar/load. In live trading you would use the
  TSO day-ahead forecasts instead (ENTSO-E Transparency Platform); using actuals as a
  proxy flatters exogenous-feature models slightly. This is stated in the README.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info"

# Energy-Charts production-type names -> our column names (values summed per column).
POWER_MAP = {
    "load": ["Load", "Load (incl. self-consumption)"],
    "wind": ["Wind onshore", "Wind offshore"],
    "solar": ["Solar"],
}


def _get(endpoint: str, params: dict, timeout: int = 60) -> dict:
    resp = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _to_index(unix_seconds: list[int]) -> pd.DatetimeIndex:
    return pd.to_datetime(unix_seconds, unit="s", utc=True)


def parse_price(payload: dict) -> pd.Series:
    s = pd.Series(payload["price"], index=_to_index(payload["unix_seconds"]), dtype="float64")
    return s.resample("h").mean().rename("price")


def parse_public_power(payload: dict) -> pd.DataFrame:
    idx = _to_index(payload["unix_seconds"])
    by_name = {pt["name"]: pt["data"] for pt in payload["production_types"]}
    out = {}
    for col, names in POWER_MAP.items():
        found = [pd.Series(by_name[n], index=idx, dtype="float64") for n in names if n in by_name]
        if not found:
            raise KeyError(f"No production type found for '{col}' among {sorted(by_name)}")
        # 'Load' and 'Load (incl. self-consumption)' are alternatives, not addends.
        out[col] = found[0] if col == "load" else sum(found)
    df = pd.DataFrame(out) / 1000.0  # MW -> GW
    return df.resample("h").mean()


def fetch(start: str, end: str, bzn: str = "DE-LU", country: str = "de") -> pd.DataFrame:
    """Download and align prices and fundamentals on an hourly UTC grid."""
    log.info("Fetching Energy-Charts data %s -> %s (%s)", start, end, bzn)
    price = parse_price(_get("price", {"bzn": bzn, "start": start, "end": end}))
    power = parse_public_power(
        _get("public_power", {"country": country, "start": start, "end": end})
    )
    df = power.join(price, how="inner")[["price", "load", "wind", "solar"]]
    df.index.name = "timestamp"
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        log.warning("Interpolating %d missing values", n_missing)
        df = df.interpolate(limit=3).dropna()
    return df
