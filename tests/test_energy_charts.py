"""Parser tests on payloads shaped like the Energy-Charts API responses (no network)."""

import numpy as np
import pandas as pd
import pytest

from powerfc.data import energy_charts
from powerfc.data.energy_charts import date_chunks, parse_price, parse_public_power

T0 = 1735686000  # 2025-01-01 00:00 CET


def test_parse_price_resamples_15min_to_hourly():
    payload = {"unix_seconds": [T0 + 900 * i for i in range(8)], "price": [10, 20, 30, 40] * 2}
    s = parse_price(payload)
    assert len(s) == 2
    np.testing.assert_allclose(s.to_numpy(), [25.0, 25.0])


def test_parse_public_power_sums_wind_and_converts_to_gw():
    ts = [T0 + 3600 * i for i in range(3)]
    payload = {
        "unix_seconds": ts,
        "production_types": [
            {"name": "Load", "data": [60000, 61000, 62000]},
            {"name": "Wind onshore", "data": [10000, 10000, 10000]},
            {"name": "Wind offshore", "data": [2000, 2000, 2000]},
            {"name": "Solar", "data": [0, 500, 1000]},
        ],
    }
    df = parse_public_power(payload)
    np.testing.assert_allclose(df["wind"], [12.0, 12.0, 12.0])
    np.testing.assert_allclose(df["load"], [60.0, 61.0, 62.0])
    np.testing.assert_allclose(df["solar"], [0.0, 0.5, 1.0])


def test_parse_public_power_missing_type_raises():
    payload = {"unix_seconds": [T0], "production_types": [{"name": "Load", "data": [1]}]}
    with pytest.raises(KeyError):
        parse_public_power(payload)


def test_date_chunks_cover_range_without_gaps():
    chunks = date_chunks("2023-01-01", "2023-03-15", days=30)
    assert chunks[0][0] == "2023-01-01"
    assert chunks[-1][1] == "2023-03-15"
    for (_, hi), (lo, _) in zip(chunks, chunks[1:], strict=False):
        assert hi == lo  # consecutive windows share their boundary
    assert len(chunks) == 3


def test_date_chunks_rejects_empty_range():
    with pytest.raises(ValueError):
        date_chunks("2023-01-02", "2023-01-01")


def test_fetch_downloads_in_chunks_and_drops_boundary_overlap(monkeypatch):
    calls = []

    def fake_get(endpoint, params, timeout=None):
        calls.append((endpoint, params["start"], params["end"]))
        lo = pd.Timestamp(params["start"], tz="UTC")
        hi = pd.Timestamp(params["end"], tz="UTC") + pd.Timedelta(hours=23)  # end day inclusive
        ts = [int(t.timestamp()) for t in pd.date_range(lo, hi, freq="h")]
        if endpoint == "price":
            return {"unix_seconds": ts, "price": [50.0] * len(ts)}
        types = [
            {"name": n, "data": [1000.0] * len(ts)}
            for n in ("Load", "Wind onshore", "Wind offshore", "Solar")
        ]
        return {"unix_seconds": ts, "production_types": types}

    monkeypatch.setattr(energy_charts, "_get", fake_get)
    df = energy_charts.fetch("2023-01-01", "2023-03-15")

    assert {e for e, _, _ in calls} == {"price", "public_power"}
    assert sum(e == "public_power" for e, _, _ in calls) == 3  # 3 windows of <= 30 days
    assert df.index.is_unique and df.index.is_monotonic_increasing
    expected = pd.date_range("2023-01-01", "2023-03-15 23:00", freq="h", tz="UTC")
    assert df.index.equals(expected.rename("timestamp"))  # no gap, no duplicate hour
    assert not df.isna().any().any()
