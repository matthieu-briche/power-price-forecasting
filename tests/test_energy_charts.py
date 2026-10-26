"""Parser tests on payloads shaped like the Energy-Charts API responses (no network)."""

import numpy as np
import pytest

from powerfc.data.energy_charts import parse_price, parse_public_power

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
