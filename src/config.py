"""Shared configuration and filesystem paths.

Every path is driven by an environment variable so that the exact same code
runs unchanged in three places:

  * locally          -> ./data, ./models
  * docker-compose   -> /opt/airflow/data, /opt/airflow/models (bind mount)
  * Kubernetes       -> /app/models (PersistentVolumeClaim, see k8s/pvc.yaml)

The Airflow container writes the model artifacts; the Flask container reads
them. Both resolve their paths through this module, so there is exactly one
place to change if the layout moves.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models")).resolve()

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_CSV = RAW_DIR / "btc_usd_raw.csv"
FEATURES_CSV = PROCESSED_DIR / "btc_features.csv"

MODEL_JSON = MODEL_DIR / "prophet_model.json"
METRICS_JSON = MODEL_DIR / "metrics.json"
FORECAST_JSON = MODEL_DIR / "forecast.json"

# Lightweight run tracking. Deliberately a directory on the PVC rather than an
# MLflow server: one less service to deploy, and the artifacts survive pod
# restarts because the PVC is mounted at /app/models.
RUNS_DIR = MODEL_DIR / "runs"

# --------------------------------------------------------------------------
# Model / data parameters
# --------------------------------------------------------------------------

TICKER = os.getenv("TICKER", "BTC-USD")

# How much history to pull from yfinance. "max" is ~2014-present for BTC-USD.
HISTORY_PERIOD = os.getenv("HISTORY_PERIOD", "5y")

# Forecast horizon in days. The assignment fixes this at 7.
FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "7"))

# Rolling windows for the engineered technical indicators (Airflow task 2).
MA_WINDOWS = (7, 30)
VOLATILITY_WINDOWS = (7, 30)

# --------------------------------------------------------------------------
# Prophet hyperparameters
#
# Chosen by a rolling-origin backtest (12 non-overlapping 7-day windows,
# ~84 days); reproduce with `python -m src.backtest`. Headline result:
#
#   full 5y history, cps=0.05, cr=0.80  ->  MAPE 18.43%
#   365d window,     cps=0.05, cr=0.95  ->  MAPE  3.49%   <- these defaults
#
# The training-window length dominates every other knob. Prophet fits one
# global piecewise-linear trend, so feeding it BTC's 2021-2023 regime biases
# the 7-day extrapolation toward a market that no longer exists. Truncating to
# the last year cut MAPE roughly five-fold on its own.
# --------------------------------------------------------------------------

# Train on only the most recent N days. 0 / unset means use all history.
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "365"))

CHANGEPOINT_PRIOR_SCALE = float(os.getenv("CHANGEPOINT_PRIOR_SCALE", "0.05"))

# Prophet's default of 0.8 places NO changepoints in the most recent 20% of
# the series -- precisely the stretch a 7-day forecast extrapolates from. 0.95
# lets the trend bend to a recent regime change.
CHANGEPOINT_RANGE = float(os.getenv("CHANGEPOINT_RANGE", "0.95"))

# BTC's swing amplitude scales with its price level rather than being a fixed
# dollar amount, so seasonality is multiplicative.
SEASONALITY_MODE = os.getenv("SEASONALITY_MODE", "multiplicative")

# Registration gate used by the Airflow retrain task: the freshly trained model
# replaces the live artifact only if its hold-out MAPE is at or below this.
#
# Set to 15% deliberately. The rolling backtest puts the tuned model at a 3.49%
# mean / 2.60% median MAPE, but a single volatile week (2026-11-13..19, BTC
# ~70k -> ~62k) scored 9.58%. A 10% gate would therefore reject a perfectly
# healthy model whenever the market moves hard. 15% clears that worst observed
# window with margin while still sitting below the untuned full-history
# baseline's 14.70% mean -- so a genuinely broken model is still caught.
MAPE_THRESHOLD = float(os.getenv("MAPE_THRESHOLD", "15.0"))

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a single stream handler at LOG_LEVEL."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False
    return logger


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
