"""Market-data providers for BTC-USD daily OHLCV.

The assignment permits yfinance, the CoinGecko API or the Binance public API.
``src/etl_fetch.py`` tries them in order and takes the first that returns a
usable frame:

  1. **yfinance** (primary). The project's chosen source. In practice Yahoo
     rate-limits aggressively and answers HTTP 429 with a body that is not
     JSON, which surfaces inside yfinance as an empty DataFrame rather than an
     exception -- so "empty" is treated as failure, not as "no data".
  2. **Binance** ``/api/v3/klines`` (failover). Full OHLCV, no API key, long
     history. Quotes BTCUSDT rather than BTC-USD; USDT holds its dollar peg
     closely enough that the close series is interchangeable for forecasting.
  3. **CoinGecko** ``/market_chart`` (last resort). Close prices only -- the
     free tier does not expose OHLC on this endpoint and caps the window at
     365 days -- so Open/High/Low are backfilled from Close and the frame is
     flagged ``synthetic_ohlc``.

Every provider returns the same contract: a DataFrame with columns
``Date, Open, High, Low, Close, Volume``, tz-naive, sorted ascending, one row
per calendar day.
"""

from __future__ import annotations

import pandas as pd
import requests

from src import config

log = config.get_logger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

HTTP_TIMEOUT = 30

# BTC-USD on Yahoo is BTCUSDT on Binance and "bitcoin" on CoinGecko.
BINANCE_SYMBOL = {"BTC-USD": "BTCUSDT"}
COINGECKO_ID = {"BTC-USD": "bitcoin"}


class ProviderError(RuntimeError):
    """Raised when a single provider fails to return usable data."""


def period_to_days(period: str) -> int:
    """Translate a yfinance-style period string ('5y', '18mo', 'max') to days."""
    period = period.strip().lower()
    if period == "max":
        return 365 * 15
    units = {"d": 1, "wk": 7, "mo": 30, "y": 365}
    for suffix in ("wk", "mo", "d", "y"):
        if period.endswith(suffix):
            try:
                return int(float(period[: -len(suffix)]) * units[suffix])
            except ValueError:
                break
    raise ValueError(f"unrecognised period string: {period!r}")


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared output contract: tz-naive dates, numeric OHLCV, sorted."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.drop_duplicates(subset="Date", keep="last")
    return df[["Date"] + REQUIRED_COLUMNS].sort_values("Date").reset_index(drop=True)


# --------------------------------------------------------------------------
# 1. yfinance
# --------------------------------------------------------------------------


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse yfinance's MultiIndex columns to plain OHLCV names.

    Recent yfinance returns ``MultiIndex([('Close', 'BTC-USD'), ...])`` even
    for a single ticker; older releases return a flat Index. Handle both rather
    than indexing positionally.
    """
    if isinstance(df.columns, pd.MultiIndex):
        levels = [set(df.columns.get_level_values(i)) for i in range(df.columns.nlevels)]
        ticker_level = next((i for i, vals in enumerate(levels) if vals == {ticker}), None)
        if ticker_level is not None:
            df.columns = df.columns.droplevel(ticker_level)
        else:
            price_level = next((i for i, vals in enumerate(levels) if "Close" in vals), 0)
            df.columns = df.columns.get_level_values(price_level)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """Pull daily bars from Yahoo Finance via yfinance."""
    import yfinance as yf

    def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
        frame = _flatten_columns(raw.copy(), ticker).reset_index()
        date_col = next(
            (c for c in ("Date", "Datetime", "index") if c in frame.columns),
            frame.columns[0],
        )
        frame = frame.rename(columns={date_col: "Date"})
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ProviderError(f"yfinance response missing columns: {missing}")
        return _finalise(frame)

    # auto_adjust=True collapses Close/Adj Close into one Close column. BTC has
    # no splits or dividends, so the adjustment is a no-op here regardless.
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is not None and not df.empty:
        return _normalise(df)

    # The Ticker object hits a different endpoint and sometimes still succeeds
    # when yf.download has been throttled down to an empty frame.
    log.debug("yf.download empty; trying Ticker.history")
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df is not None and not df.empty:
        return _normalise(df)

    raise ProviderError(
        "yfinance returned an empty frame from both endpoints "
        "(Yahoo answers HTTP 429 when rate-limiting)"
    )


# --------------------------------------------------------------------------
# 2. Binance
# --------------------------------------------------------------------------

BINANCE_URL = "https://api.binance.com/api/v3/klines"
BINANCE_MAX_LIMIT = 1000


def fetch_binance(ticker: str, period: str) -> pd.DataFrame:
    """Pull daily klines from the Binance public API, paging as needed."""
    symbol = BINANCE_SYMBOL.get(ticker, ticker.replace("-", "").replace("USD", "USDT"))
    days = period_to_days(period)

    start_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)

    rows: list[list] = []
    # The endpoint caps a response at 1000 candles, so walk forward from
    # start_time until we reach today.
    while start_ms < now_ms:
        resp = requests.get(
            BINANCE_URL,
            params={
                "symbol": symbol,
                "interval": "1d",
                "startTime": start_ms,
                "limit": BINANCE_MAX_LIMIT,
            },
            timeout=HTTP_TIMEOUT,
        )
        if resp.status_code != 200:
            raise ProviderError(f"binance HTTP {resp.status_code}: {resp.text[:200]}")

        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)

        last_open_ms = batch[-1][0]
        if len(batch) < BINANCE_MAX_LIMIT:
            break
        start_ms = last_open_ms + 86_400_000  # advance one day past the last candle

    if not rows:
        raise ProviderError(f"binance returned no klines for {symbol}")

    # Kline layout: [openTime, open, high, low, close, volume, closeTime, ...]
    df = pd.DataFrame(
        [
            {
                "Date": pd.to_datetime(r[0], unit="ms", utc=True),
                "Open": r[1],
                "High": r[2],
                "Low": r[3],
                "Close": r[4],
                "Volume": r[5],
            }
            for r in rows
        ]
    )
    log.info("binance: %d daily candles for %s", len(df), symbol)
    return _finalise(df)


# --------------------------------------------------------------------------
# 3. CoinGecko
# --------------------------------------------------------------------------

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
COINGECKO_FREE_MAX_DAYS = 365


def fetch_coingecko(ticker: str, period: str) -> pd.DataFrame:
    """Pull daily close prices from CoinGecko. Close-only -- OHLC is synthesised."""
    coin = COINGECKO_ID.get(ticker, "bitcoin")
    days = min(period_to_days(period), COINGECKO_FREE_MAX_DAYS)
    if period_to_days(period) > COINGECKO_FREE_MAX_DAYS:
        log.warning(
            "coingecko free tier caps history at %d days; requested %s",
            COINGECKO_FREE_MAX_DAYS,
            period,
        )

    resp = requests.get(
        COINGECKO_URL.format(coin=coin),
        params={"vs_currency": "usd", "days": days, "interval": "daily"},
        timeout=HTTP_TIMEOUT,
    )
    if resp.status_code != 200:
        raise ProviderError(f"coingecko HTTP {resp.status_code}: {resp.text[:200]}")

    payload = resp.json()
    prices = payload.get("prices") or []
    volumes = {int(ts): vol for ts, vol in payload.get("total_volumes") or []}
    if not prices:
        raise ProviderError("coingecko returned no price series")

    # This endpoint exposes close only. Open/High/Low are backfilled from Close
    # so the frame satisfies the shared contract -- downstream code must not
    # read intraday range off a CoinGecko-sourced frame.
    df = pd.DataFrame(
        [
            {
                "Date": pd.to_datetime(int(ts), unit="ms", utc=True),
                "Open": price,
                "High": price,
                "Low": price,
                "Close": price,
                "Volume": volumes.get(int(ts), 0.0),
            }
            for ts, price in prices
        ]
    )
    log.warning("coingecko: %d rows, OHLC synthesised from Close", len(df))
    return _finalise(df)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

PROVIDERS = {
    "yfinance": fetch_yfinance,
    "binance": fetch_binance,
    "coingecko": fetch_coingecko,
}

DEFAULT_PROVIDER_ORDER = ("yfinance", "binance", "coingecko")
