"""MongoDB persistence for market data, forecasts and back-test runs.

Collections
-----------
market_data  one document per (market, timestamp): price + fundamentals. Idempotent upserts,
             so re-ingesting an overlapping period never creates duplicates.
forecasts    one document per (run_id, model, timestamp).
runs         one document per back-test: config, metrics, code version, timestamps.
"""

from __future__ import annotations

import datetime as dt
import uuid

import pandas as pd
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.database import Database

from powerfc import COLUMNS, __version__


class MongoStore:
    def __init__(self, uri: str | None = None, db_name: str = "powerfc", client=None):
        self.client = client or MongoClient(uri or "mongodb://localhost:27017")
        self.db: Database = self.client[db_name]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.db.market_data.create_index(
            [("market", ASCENDING), ("timestamp", ASCENDING)], unique=True
        )
        self.db.forecasts.create_index(
            [("run_id", ASCENDING), ("model", ASCENDING), ("timestamp", ASCENDING)], unique=True
        )
        self.db.runs.create_index([("created_at", ASCENDING)])

    # ---- market data -------------------------------------------------------------------
    def upsert_market_data(self, df: pd.DataFrame, market: str = "DE-LU") -> int:
        ops = []
        for ts, row in df[COLUMNS].iterrows():
            key = {"market": market, "timestamp": ts.to_pydatetime()}
            ops.append(UpdateOne(key, {"$set": {**key, **row.to_dict()}}, upsert=True))
        if not ops:
            return 0
        res = self.db.market_data.bulk_write(ops, ordered=False)
        return res.upserted_count + res.modified_count

    def load_market_data(self, market: str = "DE-LU", start=None, end=None) -> pd.DataFrame:
        q: dict = {"market": market}
        if start or end:
            q["timestamp"] = {}
            if start:
                q["timestamp"]["$gte"] = pd.Timestamp(start, tz="UTC").to_pydatetime()
            if end:
                q["timestamp"]["$lt"] = pd.Timestamp(end, tz="UTC").to_pydatetime()
        docs = list(self.db.market_data.find(q, {"_id": 0}).sort("timestamp", ASCENDING))
        if not docs:
            return pd.DataFrame(columns=COLUMNS)
        df = pd.DataFrame(docs)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")[COLUMNS]

    # ---- back-test runs ----------------------------------------------------------------
    def save_run(
        self, preds: pd.DataFrame, metrics: pd.DataFrame, config: dict, market: str
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        self.db.runs.insert_one(
            {
                "run_id": run_id,
                "created_at": dt.datetime.now(dt.UTC),
                "market": market,
                "code_version": __version__,
                "config": config,
                "metrics": metrics.reset_index().to_dict("records"),
            }
        )
        models = [c for c in preds.columns if c not in ("date", "y")]
        docs = [
            {
                "run_id": run_id,
                "model": m,
                "timestamp": ts.to_pydatetime(),
                "y_true": float(row["y"]),
                "y_pred": float(row[m]),
            }
            for ts, row in preds.iterrows()
            for m in models
        ]
        if docs:
            self.db.forecasts.insert_many(docs)
        return run_id

    def mae_by_model_and_hour(self, run_id: str) -> pd.DataFrame:
        """Server-side aggregation: MAE per model and UTC hour for one run."""
        pipeline = [
            {"$match": {"run_id": run_id}},
            {
                "$group": {
                    "_id": {"model": "$model", "hour": {"$hour": "$timestamp"}},
                    "mae": {"$avg": {"$abs": {"$subtract": ["$y_true", "$y_pred"]}}},
                    "n": {"$sum": 1},
                }
            },
            {"$sort": {"_id.model": 1, "_id.hour": 1}},
        ]
        rows = [
            {"model": r["_id"]["model"], "hour_utc": r["_id"]["hour"], "mae": r["mae"], "n": r["n"]}
            for r in self.db.forecasts.aggregate(pipeline)
        ]
        return pd.DataFrame(rows)
