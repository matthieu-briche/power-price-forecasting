"""Command-line entry point: ``powerfc ingest`` and ``powerfc backtest``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from powerfc.backtest import BacktestConfig, config_dict, run_backtest

log = logging.getLogger("powerfc")


def _load_source(args) -> pd.DataFrame:
    if args.source == "synthetic":
        from powerfc.data import generate_synthetic

        return generate_synthetic(start=args.start, days=args.days, seed=args.seed)
    if args.source == "energy-charts":
        from powerfc.data.energy_charts import fetch

        end = (
            args.end or (pd.Timestamp(args.start) + pd.Timedelta(days=args.days)).date().isoformat()
        )
        return fetch(args.start, end, bzn=args.market)
    if args.source == "csv":
        df = pd.read_csv(args.path, index_col="timestamp", parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    if args.source == "mongo":
        from powerfc.storage import MongoStore

        return MongoStore(args.mongo_uri).load_market_data(args.market)
    raise ValueError(args.source)


def cmd_ingest(args) -> int:
    df = _load_source(args)
    log.info("Loaded %d hourly rows (%s -> %s)", len(df), df.index.min(), df.index.max())
    if args.mongo_uri:
        from powerfc.storage import MongoStore

        n = MongoStore(args.mongo_uri).upsert_market_data(df, market=args.market)
        log.info("Upserted %d documents into MongoDB", n)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        log.info("Wrote %s", args.out)
    return 0


def cmd_backtest(args) -> int:
    df = _load_source(args)
    cfg = BacktestConfig(
        models=tuple(args.models),
        test_days=args.test_days,
        train_window_days=args.train_window,
        retrain_every=args.retrain_every,
    )
    preds, metrics = run_backtest(df, cfg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv")
    metrics.to_csv(out / "metrics.csv")
    (out / "config.json").write_text(json.dumps(config_dict(cfg), indent=2, default=str))
    print(metrics.to_string())

    if args.mongo_uri:
        from powerfc.storage import MongoStore

        run_id = MongoStore(args.mongo_uri).save_run(preds, metrics, config_dict(cfg), args.market)
        log.info("Saved run %s to MongoDB", run_id)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="powerfc", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def add_source(sp):
        sp.add_argument(
            "--source", choices=["synthetic", "energy-charts", "csv", "mongo"], default="synthetic"
        )
        sp.add_argument("--start", default="2023-01-01")
        sp.add_argument("--end", default=None)
        sp.add_argument("--days", type=int, default=730)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--path", help="CSV path for --source csv")
        sp.add_argument("--market", default="DE-LU")
        sp.add_argument("--mongo-uri", default=None)

    ing = sub.add_parser("ingest", help="Load market data and store it (MongoDB and/or CSV)")
    add_source(ing)
    ing.add_argument("--out", default=None, help="Optional CSV output path")
    ing.set_defaults(func=cmd_ingest)

    bt = sub.add_parser("backtest", help="Walk-forward back-test of forecasting models")
    add_source(bt)
    bt.add_argument("--models", nargs="+", default=["naive", "ridge", "hgb"])
    bt.add_argument("--test-days", type=int, default=90)
    bt.add_argument("--train-window", type=int, default=365)
    bt.add_argument("--retrain-every", type=int, default=7)
    bt.add_argument("--out-dir", default="reports")
    bt.set_defaults(func=cmd_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
