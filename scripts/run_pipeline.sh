#!/usr/bin/env bash
# End-to-end pipeline: ingest market data into MongoDB, then back-test from MongoDB.
#
#   MONGO_URI   MongoDB connection string   (default mongodb://localhost:27017)
#   SOURCE      synthetic | energy-charts   (default synthetic)
#   START/DAYS  history to ingest           (default 2023-01-01 / 730)
#   TEST_DAYS   back-test length in days    (default 90)
set -euo pipefail

MONGO_URI="${MONGO_URI:-mongodb://localhost:27017}"
SOURCE="${SOURCE:-synthetic}"
START="${START:-2023-01-01}"
DAYS="${DAYS:-730}"
TEST_DAYS="${TEST_DAYS:-90}"
OUT_DIR="${OUT_DIR:-reports}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "Ingesting ${DAYS} days from '${SOURCE}' starting ${START}"
powerfc ingest --source "${SOURCE}" --start "${START}" --days "${DAYS}" --mongo-uri "${MONGO_URI}"

log "Running walk-forward back-test (${TEST_DAYS} days) from MongoDB"
powerfc backtest --source mongo --mongo-uri "${MONGO_URI}" \
  --test-days "${TEST_DAYS}" --out-dir "${OUT_DIR}"

log "Done. Results in ${OUT_DIR}/ and in the 'runs' / 'forecasts' MongoDB collections."
