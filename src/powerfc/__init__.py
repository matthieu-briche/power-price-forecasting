"""powerfc: day-ahead electricity price forecasting and back-testing."""

__version__ = "0.1.0"

# Canonical hourly dataset schema used across the package.
# Index: tz-aware hourly DatetimeIndex (UTC). Columns below, prices in EUR/MWh, volumes in GW.
COLUMNS = ["price", "load", "wind", "solar"]
