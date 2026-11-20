"""Pull daily BTC-USD OHLCV data into raw storage.

Part C1, task 1 of the Airflow DAG. Responsibilities:

  * Try each configured provider in turn (see ``src/providers.py``), retrying
    each with exponential backoff + jitter before failing over to the next.
  * Drop today's partial candle -- an in-progress day has an incomplete close
    and would otherwise be trained on as if it were a settled price.
  * Merge the result into a cached raw CSV so that a later rate-limited run
    still has yesterday's history to fall back on.

Usage:
    python -m src.etl_fetch
    python -m src.etl_fetch --period max --providers binance,yfinance
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd

from src import config, providers

log = config.get_logger(__name__)

MAX_ATTEMPTS_PER_PROVIDER = int(os.getenv("FETCH_MAX_ATTEMPTS", "3"))
BASE_BACKOFF_SECONDS = 2.0


class DataFetchError(RuntimeError):
    """Raised when market data could not be obtained from any provider."""


def _provider_order() -> tuple[str, ...]:
    """Resolve the provider chain from the DATA_PROVIDERS env var."""
    raw = os.getenv("DATA_PROVIDERS")
    if not raw:
        return providers.DEFAULT_PROVIDER_ORDER
    chain = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    unknown = [p for p in chain if p not in providers.PROVIDERS]
    if unknown:
        raise ValueError(f"unknown provider(s) {unknown}; choose from {list(providers.PROVIDERS)}")
    return chain


def drop_partial_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Remove today's still-open daily candle.

    Every provider returns the current UTC day as a partial bar whose close is
    just the last trade, not a settled daily close. Training on it teaches the
    model a price that will still move for hours.
    """
    if df.empty:
        return df
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    partial = df["Date"] >= today
    if partial.any():
        log.info("Dropping %d partial candle(s) for %s", int(partial.sum()), today.date())
        df = df[~partial].reset_index(drop=True)
    return df


def fetch_btc_data(
    ticker: str = config.TICKER,
    period: str = config.HISTORY_PERIOD,
    provider_chain: tuple[str, ...] | None = None,
    max_attempts: int = MAX_ATTEMPTS_PER_PROVIDER,
) -> tuple[pd.DataFrame, str]:
    """Fetch daily OHLCV, failing over across providers. Returns (frame, provider)."""
    chain = provider_chain or _provider_order()
    errors: dict[str, str] = {}

    for provider_name in chain:
        fetch_fn = providers.PROVIDERS[provider_name]

        for attempt in range(1, max_attempts + 1):
            try:
                log.info(
                    "Fetching %s from %s (period=%s), attempt %d/%d",
                    ticker,
                    provider_name,
                    period,
                    attempt,
                    max_attempts,
                )
                df = drop_partial_candle(fetch_fn(ticker, period))
                if df.empty:
                    raise providers.ProviderError("provider returned no settled candles")

                log.info(
                    "%s: %d rows, %s -> %s",
                    provider_name,
                    len(df),
                    df["Date"].min().date(),
                    df["Date"].max().date(),
                )
                return df, provider_name

            except Exception as exc:  # noqa: BLE001 - any failure is retryable
                errors[provider_name] = str(exc)
                if attempt == max_attempts:
                    log.warning(
                        "%s exhausted after %d attempts (%s); failing over",
                        provider_name,
                        max_attempts,
                        exc,
                    )
                    break
                # Jitter stops concurrent Airflow task retries synchronising.
                delay = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                log.warning(
                    "%s attempt %d failed (%s); retrying in %.1fs",
                    provider_name,
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)

    detail = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
    raise DataFetchError(f"all providers failed -> {detail}")


def merge_with_cache(fresh: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
    """Union a fresh pull with the cached CSV, the newer value winning per date."""
    if not cache_path.exists():
        return fresh

    try:
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
    except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
        log.warning("Could not read cache at %s (%s); ignoring it", cache_path, exc)
        return fresh

    cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.drop_duplicates(subset="Date", keep="last")
    combined = combined.sort_values("Date").reset_index(drop=True)

    log.info(
        "Merged with cache: %d cached + %d fetched -> %d rows (%d new)",
        len(cached),
        len(fresh),
        len(combined),
        max(len(combined) - len(cached), 0),
    )
    return combined


def load_cache(cache_path: Path) -> pd.DataFrame:
    """Read the cached raw CSV, or raise if there is nothing to fall back on."""
    if not cache_path.exists():
        raise DataFetchError(f"fetch failed and no cache exists at {cache_path}; cannot proceed")
    df = pd.read_csv(cache_path, parse_dates=["Date"])
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    log.warning("Serving %d rows from stale cache at %s", len(df), cache_path)
    return df


def run(
    ticker: str = config.TICKER,
    period: str = config.HISTORY_PERIOD,
    out_path: Path | None = None,
    provider_chain: tuple[str, ...] | None = None,
) -> Path:
    """Fetch, merge with cache and persist. Entry point for the Airflow task."""
    config.ensure_dirs()
    out_path = Path(out_path) if out_path else config.RAW_CSV

    try:
        fresh, provider_name = fetch_btc_data(ticker, period, provider_chain)
        combined = merge_with_cache(fresh, out_path)
        log.info("Source for this run: %s", provider_name)
    except DataFetchError as exc:
        # A rate-limited run must not wipe out good history. Degrade to the
        # cache so downstream training still has something to work with.
        log.error("Live fetch failed: %s", exc)
        combined = load_cache(out_path)

    combined.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(combined), out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch daily BTC-USD data into raw storage.")
    parser.add_argument("--ticker", default=config.TICKER)
    parser.add_argument(
        "--period", default=config.HISTORY_PERIOD, help="period string, e.g. 1y, 5y, max"
    )
    parser.add_argument("--out", default=None, help="output CSV path")
    parser.add_argument(
        "--providers", default=None, help=f"comma-separated chain from {list(providers.PROVIDERS)}"
    )
    args = parser.parse_args(argv)

    chain = None
    if args.providers:
        chain = tuple(p.strip().lower() for p in args.providers.split(",") if p.strip())

    try:
        path = run(ticker=args.ticker, period=args.period, out_path=args.out, provider_chain=chain)
    except DataFetchError as exc:
        log.error("ETL failed: %s", exc)
        return 1

    df = pd.read_csv(path)
    print(f"\nRaw data written to {path}")
    print(f"Rows: {len(df)}  |  Range: {df['Date'].min()} -> {df['Date'].max()}")
    print(df.tail(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
