"""Rolling-origin backtest for the Prophet forecaster.

The graded metric (Part A3 / C1) is a single hold-out of the final 7 days, and
that is what ``src/train_forecast.py`` registers. A single window is a noisy
estimate though -- it can flatter or punish a config by luck. This module
re-scores the same config across several consecutive non-overlapping 7-day
windows and reports the distribution, which is what the default hyperparameters
in ``src/config.py`` were actually chosen on.

Usage:
    python -m src.backtest
    python -m src.backtest --origins 12 --train-window-days 365
    python -m src.backtest --compare          # defaults vs. full-history
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from src import config, train_forecast

log = config.get_logger(__name__)

DEFAULT_ORIGINS = 12


def _quiet_stan() -> None:
    """Silence cmdstanpy's per-fit chatter; a backtest runs dozens of fits."""
    warnings.filterwarnings("ignore")
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def rolling_backtest(
    df: pd.DataFrame,
    horizon: int = config.FORECAST_DAYS,
    origins: int = DEFAULT_ORIGINS,
    train_window_days: int = config.TRAIN_WINDOW_DAYS,
    **model_kwargs,
) -> pd.DataFrame:
    """Score ``origins`` consecutive non-overlapping ``horizon``-day windows.

    Walks the cut point backwards from the end of the series. Each iteration
    trains only on data strictly before its cut, so there is no leakage.
    """
    _quiet_stan()
    rows = []

    for k in range(origins, 0, -1):
        cut = len(df) - k * horizon
        if cut < horizon * 10:
            log.warning("Skipping origin -%d: only %d training rows available", k, cut)
            continue

        train_df = train_forecast.apply_train_window(df.iloc[:cut], train_window_days)
        test_df = df.iloc[cut : cut + horizon]
        if len(test_df) < horizon:
            continue

        _, forecast = train_forecast.fit_and_forecast(train_df, horizon, (), **model_kwargs)
        metrics = train_forecast.compute_metrics(
            test_df["y"].to_numpy(), forecast["yhat"].to_numpy()
        )
        rows.append(
            {
                "origin": str(train_df["ds"].max().date()),
                "test_start": str(test_df["ds"].min().date()),
                "test_end": str(test_df["ds"].max().date()),
                "train_rows": len(train_df),
                **metrics,
            }
        )
        log.info(
            "origin %s -> MAE %8.0f  RMSE %8.0f  MAPE %6.2f%%",
            rows[-1]["origin"],
            metrics["mae"],
            metrics["rmse"],
            metrics["mape"],
        )

    if not rows:
        raise ValueError("no usable backtest origins; the series is too short")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> dict[str, float]:
    """Collapse per-window results into headline numbers."""
    return {
        "windows": int(len(results)),
        "mae_mean": round(float(results["mae"].mean()), 2),
        "rmse_mean": round(float(results["rmse"].mean()), 2),
        "mape_mean": round(float(results["mape"].mean()), 3),
        "mape_median": round(float(results["mape"].median()), 3),
        "mape_worst": round(float(results["mape"].max()), 3),
        "mape_best": round(float(results["mape"].min()), 3),
    }


def _print_report(title: str, results: pd.DataFrame, summary: dict) -> None:
    print(f"\n{title}")
    print("=" * 72)
    print(f"  {'TEST WINDOW':<26} {'ROWS':>7} {'MAE':>10} {'RMSE':>10} {'MAPE %':>9}")
    print("  " + "-" * 66)
    for r in results.itertuples():
        print(
            f"  {r.test_start} -> {r.test_end:<12} {r.train_rows:>7} "
            f"{r.mae:>10,.0f} {r.rmse:>10,.0f} {r.mape:>9.2f}"
        )
    print("  " + "-" * 66)
    print(
        f"  {'MEAN':<26} {'':>7} {summary['mae_mean']:>10,.0f} "
        f"{summary['rmse_mean']:>10,.0f} {summary['mape_mean']:>9.2f}"
    )
    print(f"  {'MEDIAN MAPE':<26} {summary['mape_median']:>38.2f}")
    print(
        f"  {'BEST / WORST MAPE':<26} "
        f"{summary['mape_best']:>30.2f} / {summary['mape_worst']:.2f}"
    )
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling-origin backtest of the BTC forecaster.")
    parser.add_argument("--features", default=None)
    parser.add_argument("--horizon", type=int, default=config.FORECAST_DAYS)
    parser.add_argument("--origins", type=int, default=DEFAULT_ORIGINS)
    parser.add_argument("--train-window-days", type=int, default=config.TRAIN_WINDOW_DAYS)
    parser.add_argument(
        "--compare", action="store_true", help="also score the untuned full-history baseline"
    )
    parser.add_argument("--out", default=None, help="write per-window results to CSV")
    args = parser.parse_args(argv)

    path = Path(args.features) if args.features else config.FEATURES_CSV
    if not path.exists():
        log.error("feature file not found at %s; run src.features first", path)
        return 1
    df = pd.read_csv(path, parse_dates=["ds"])[["ds", "y"]]

    tuned = rolling_backtest(df, args.horizon, args.origins, args.train_window_days)
    tuned_summary = summarise(tuned)
    _print_report(
        f"TUNED  (window={args.train_window_days}d, cps={config.CHANGEPOINT_PRIOR_SCALE}, "
        f"cr={config.CHANGEPOINT_RANGE}, {config.SEASONALITY_MODE})",
        tuned,
        tuned_summary,
    )

    if args.compare:
        base = rolling_backtest(
            df,
            args.horizon,
            args.origins,
            train_window_days=0,
            changepoint_range=0.8,
        )
        base_summary = summarise(base)
        _print_report("BASELINE  (full history, cr=0.80, Prophet defaults)", base, base_summary)

        lift = base_summary["mape_mean"] - tuned_summary["mape_mean"]
        factor = base_summary["mape_mean"] / max(tuned_summary["mape_mean"], 1e-9)
        print(
            f"\n  Tuning improved mean MAPE by {lift:.2f} points "
            f"({base_summary['mape_mean']:.2f}% -> {tuned_summary['mape_mean']:.2f}%, "
            f"{factor:.1f}x better)\n"
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tuned.to_csv(out, index=False)
        print(f"  Per-window results written to {out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
