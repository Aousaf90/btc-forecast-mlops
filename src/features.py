"""Feature engineering for the BTC forecasting pipeline.

Backs the Airflow ``preprocess_features`` task (Part C1, step 2): cleans the
raw OHLCV frame, computes the technical indicators the rubric asks for
(7-/30-day moving averages and volatility) and emits a frame in the exact
``ds``/``y`` shape Prophet requires.

A note on the indicators, because this is a real modelling trap:

    Prophet's ``add_regressor`` needs a value for every FUTURE date being
    predicted. A 7-day moving average over the next 7 days is not knowable at
    prediction time -- computing it would require the very prices we are
    forecasting. So these columns are computed and persisted (the rubric
    requires that, and they are genuinely useful for the report's EDA), but
    they are NOT wired into ``add_regressor`` by default. See
    ``src/train_forecast.py`` for the opt-in path and its caveat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

log = config.get_logger(__name__)


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad rows, enforce a continuous daily index and forward-fill gaps.

    BTC trades every day, so unlike equities a missing calendar day is a data
    gap rather than a weekend. Reindexing onto a continuous daily range and
    forward-filling keeps the rolling windows honest.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates(subset="Date", keep="last").sort_values("Date")

    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    gaps = len(full_range) - len(df)
    if gaps:
        log.info("Reindexing onto a continuous daily range: filling %d missing day(s)", gaps)

    df = df.set_index("Date").reindex(full_range).ffill()
    df.index.name = "Date"
    return df.reset_index()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append moving averages, volatility and returns to a cleaned frame."""
    df = df.copy()

    # Daily log return: the basis for the volatility estimate below.
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    for window in config.MA_WINDOWS:
        df[f"ma_{window}"] = df["Close"].rolling(window=window, min_periods=window).mean()

    for window in config.VOLATILITY_WINDOWS:
        # Annualised rolling standard deviation of daily log returns. Crypto
        # trades 365 days a year, so scale by sqrt(365) rather than sqrt(252).
        df[f"volatility_{window}"] = df["log_return"].rolling(
            window=window, min_periods=window
        ).std() * np.sqrt(365)

    # Momentum-style context features, cheap to compute and useful in the report.
    df["ma_ratio_7_30"] = df["ma_7"] / df["ma_30"]
    df["pct_change_1d"] = df["Close"].pct_change() * 100

    return df


def to_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to Prophet's required ``ds`` / ``y`` columns, keeping indicators."""
    out = df.rename(columns={"Date": "ds", "Close": "y"})

    indicator_cols = (
        [f"ma_{w}" for w in config.MA_WINDOWS]
        + [f"volatility_{w}" for w in config.VOLATILITY_WINDOWS]
        + ["ma_ratio_7_30", "pct_change_1d", "log_return"]
    )
    keep = ["ds", "y", "Open", "High", "Low", "Volume"] + indicator_cols
    keep = [c for c in keep if c in out.columns]

    out = out[keep]
    # Prophet is strict: ds must be datetime64 and y must be float.
    out["ds"] = pd.to_datetime(out["ds"])
    out["y"] = out["y"].astype(float)
    return out.reset_index(drop=True)


def build_features(raw_df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Full raw -> Prophet-ready pipeline.

    Args:
        raw_df: frame with Date + OHLCV columns, as written by src/etl_fetch.
        drop_warmup: drop the leading rows where the 30-day windows are still
            NaN. Prophet itself only needs ds/y, but dropping the warm-up keeps
            the persisted feature file free of nulls.
    """
    cleaned = clean_raw(raw_df)
    enriched = add_indicators(cleaned)
    prophet_df = to_prophet_frame(enriched)

    if drop_warmup:
        longest_window = max(max(config.MA_WINDOWS), max(config.VOLATILITY_WINDOWS))
        before = len(prophet_df)
        prophet_df = prophet_df.iloc[longest_window:].reset_index(drop=True)
        log.info(
            "Dropped %d warm-up rows for the %d-day windows",
            before - len(prophet_df),
            longest_window,
        )

    return prophet_df


def run(raw_path: Path | None = None, out_path: Path | None = None) -> Path:
    """Read raw CSV, build features, persist. Entry point for the Airflow task."""
    config.ensure_dirs()
    raw_path = Path(raw_path) if raw_path else config.RAW_CSV
    out_path = Path(out_path) if out_path else config.FEATURES_CSV

    if not raw_path.exists():
        raise FileNotFoundError(f"raw data not found at {raw_path}; run src.etl_fetch first")

    raw = pd.read_csv(raw_path, parse_dates=["Date"])
    features = build_features(raw)

    if features.empty:
        raise ValueError("feature engineering produced an empty frame")

    features.to_csv(out_path, index=False)
    log.info("Wrote %d feature rows to %s", len(features), out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Prophet-ready features from raw BTC data.")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    path = run(raw_path=args.raw, out_path=args.out)

    df = pd.read_csv(path)
    print(f"\nFeatures written to {path}")
    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(df.tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
