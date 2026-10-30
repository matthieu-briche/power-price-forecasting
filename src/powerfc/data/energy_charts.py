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
* Long ranges are downloaded in chunks (``CHUNK_DAYS``) with retries: a single
  multi-year ``public_power`` request is too large for the API and times out.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info"
CHUNK_DAYS = 30
TIMEOUT_S = 120

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    """Shared session retrying transient failures (timeouts, 429, 5xx) with backoff."""
    global _session
    if _session is None:
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        _session = requests.Session()
        _session.mount("https://", HTTPAdapter(max_retries=retry))
    return _session


# Energy-Charts production-type names -> our column names (values summed per column).
POWER_MAP = {
    "load": ["Load", "Load (incl. self-consumption)"],
    "wind": ["Wind onshore", "Wind offshore"],
    "solar": ["Solar"],
}


def _get(endpoint: str, params: dict, timeout: int = TIMEOUT_S) -> dict:
    resp = _get_session().get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def date_chunks(start: str, end: str, days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    """Split [start, end] into consecutive windows of at most ``days`` days.

    Consecutive windows share their boundary date; the overlap is removed after download.
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if hi <= lo:
        raise ValueError(f"end ({end}) must be after start ({start})")
    out = []
    while lo < hi:
        nxt = min(lo + pd.Timedelta(days=days), hi)
        out.append((lo.date().isoformat(), nxt.date().isoformat()))
        lo = nxt
    return out


def _fetch_chunked(endpoint: str, params: dict, parse, start: str, end: str):
    """Download ``endpoint`` window by window, parse each payload and stitch the result."""
    chunks = date_chunks(start, end)
    parts = []
    for i, (lo, hi) in enumerate(chunks, 1):
        log.info("  %s %d/%d: %s -> %s", endpoint, i, len(chunks), lo, hi)
        parts.append(parse(_get(endpoint, {**params, "start": lo, "end": hi})))
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="first")]


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
    price = _fetch_chunked("price", {"bzn": bzn}, parse_price, start, end)
    power = _fetch_chunked("public_power", {"country": country}, parse_public_power, start, end)
    df = power.join(price, how="inner")[["price", "load", "wind", "solar"]]
    df.index.name = "timestamp"
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        log.warning("Interpolating %d missing values", n_missing)
        df = df.interpolate(limit=3).dropna()
    return df
