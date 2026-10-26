"""Render the README figure from a back-test output folder.

Usage: python scripts/plot_report.py reports/   (requires matplotlib)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
SERIES = {"actual": INK, "hgb": "#2a78d6", "naive": "#eb6834", "ridge": "#1baf7a"}


def main(folder: str) -> None:
    out = Path(folder)
    preds = pd.read_csv(out / "predictions.csv", index_col="timestamp", parse_dates=True)
    metrics = pd.read_csv(out / "metrics.csv", index_col="model")
    preds.index = preds.index.tz_convert("Europe/Berlin")

    week = preds.loc[preds.index >= preds.index.max() - pd.Timedelta(days=7)]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2.6, 1]}, dpi=150
    )

    ax1.plot(week.index, week["y"], color=SERIES["actual"], lw=2, label="Actual")
    ax1.plot(week.index, week["naive"], color=SERIES["naive"], lw=1.6, ls="--", label="Naive")
    ax1.plot(week.index, week["hgb"], color=SERIES["hgb"], lw=2, label="Gradient boosting")
    ax1.set_title("Last back-test week: day-ahead price, EUR/MWh", loc="left", color=INK)
    ax1.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=week.index.tz))

    order = metrics.sort_values("mae", ascending=False)
    colors = [SERIES.get(m, MUTED) for m in order.index]
    bars = ax2.barh(order.index, order["mae"], color=colors, height=0.55)
    ax2.bar_label(bars, labels=[f"{v:.1f}" for v in order["mae"]], padding=4, color=INK)
    ax2.set_title("MAE over the test period, EUR/MWh", loc="left", color=INK)
    ax2.set_xlim(0, order["mae"].max() * 1.2)

    for ax in (ax1, ax2):
        ax.grid(axis="x" if ax is ax2 else "y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)

    fig.tight_layout()
    fig.savefig(out / "backtest.png", facecolor="white")
    print(f"Wrote {out / 'backtest.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports")
