"""Train the 7-day BTC close-price forecasting model and save artifacts.

Protocol
--------
1. Hold out the most recent ``FORECAST_DAYS`` (7) days as a test window.
2. Fit Prophet on everything before that window and score the 7-day forecast
   against it -> MAE / RMSE / MAPE. These are the numbers /metrics serves.
3. If the hold-out MAPE passes ``MAPE_THRESHOLD``, refit on the FULL history
   and persist that model. Refitting on all data is what makes the live
   forecast start from today rather than from a week ago; the hold-out fit
   exists purely to produce an honest estimate of error.
4. Serialize with ``prophet.serialize.model_to_json``.

Serialization note: Prophet models are NOT picklable in a version-stable way --
the fitted Stan backend does not round-trip reliably across releases. Since the
Airflow container writes the artifact that the Flask container reads, pickle
would be a live failure mode. ``model_to_json`` is the supported path and is
what both sides use.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from src import config, features

log = config.get_logger(__name__)

# Indicator columns that MAY be used as Prophet regressors. Off by default --
# see the caveat in the add_regressor block of build_model().
CANDIDATE_REGRESSORS = ("ma_7", "ma_30", "volatility_7", "volatility_30")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and MAPE for a forecast window."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty window")

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # Guard against a zero actual. BTC close is never 0, but a divide-by-zero
    # here would silently poison the registration gate.
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs(errors / denom)) * 100.0)

    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def build_model(
    changepoint_prior_scale: float = config.CHANGEPOINT_PRIOR_SCALE,
    changepoint_range: float = config.CHANGEPOINT_RANGE,
    seasonality_mode: str = config.SEASONALITY_MODE,
    interval_width: float = 0.95,
    regressors: tuple[str, ...] = (),
) -> Prophet:
    """Construct an unfitted Prophet model.

    Defaults come from the rolling-origin backtest documented in src/config.py
    (run `python -m src.backtest` to reproduce):
      * weekly + yearly seasonality on, daily off -- we only have daily bars.
      * multiplicative seasonality: BTC's swing amplitude scales with price.
      * changepoint_range 0.95 rather than Prophet's 0.8, so the trend can bend
        inside the recent window a 7-day forecast extrapolates from.
    """
    model = Prophet(
        growth="linear",
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=changepoint_range,
        seasonality_mode=seasonality_mode,
        interval_width=interval_width,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    # CAVEAT: enabling regressors requires supplying their values for the
    # future dates too. A 7-day moving average over the next 7 days depends on
    # the prices we are trying to predict, so it has to be extrapolated --
    # which leaks a guess into the model. Left off by default; see
    # extend_regressors() for the naive hold-last-value extrapolation used when
    # this is switched on deliberately.
    for name in regressors:
        model.add_regressor(name, standardize=True)

    return model


def extend_regressors(
    history: pd.DataFrame,
    future: pd.DataFrame,
    regressors: tuple[str, ...],
) -> pd.DataFrame:
    """Fill regressor values for future dates by holding the last value flat.

    This is the honest-but-crude option: a flat carry-forward of the final
    observed moving average / volatility. It is NOT a real forecast of those
    indicators, and the resulting model is only as good as that assumption.
    """
    future = future.copy()
    for name in regressors:
        last_value = history[name].iloc[-1]
        merged = future[["ds"]].merge(history[["ds", name]], on="ds", how="left")
        future[name] = merged[name].ffill().fillna(last_value).to_numpy()
    return future


def fit_and_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    regressors: tuple[str, ...] = (),
    **model_kwargs,
) -> tuple[Prophet, pd.DataFrame]:
    """Fit Prophet on ``train_df`` and forecast ``horizon`` days ahead."""
    model = build_model(regressors=regressors, **model_kwargs)

    fit_cols = ["ds", "y"] + list(regressors)
    model.fit(train_df[fit_cols])

    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    if regressors:
        future = extend_regressors(train_df, future, regressors)

    forecast = model.predict(future)
    return model, forecast


# --------------------------------------------------------------------------
# Artifact I/O
# --------------------------------------------------------------------------


def save_model(model: Prophet, path: Path) -> None:
    """Serialize via Prophet's own JSON format (never pickle -- see module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Write to a temp file then rename, so a Flask pod calling /reload can
    # never observe a half-written model.
    tmp.write_text(model_to_json(model), encoding="utf-8")
    tmp.replace(path)
    log.info("Saved model to %s (%.1f KB)", path, path.stat().st_size / 1024)


def load_model(path: Path) -> Prophet:
    """Deserialize a Prophet model written by save_model()."""
    return model_from_json(Path(path).read_text(encoding="utf-8"))


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def forecast_to_records(forecast: pd.DataFrame) -> list[dict]:
    """Shape a Prophet forecast frame into the JSON the API returns."""
    cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
    out = forecast[cols].copy()
    return [
        {
            "date": row.ds.strftime("%Y-%m-%d"),
            "predicted_close": round(float(row.yhat), 2),
            "lower_bound": round(float(row.yhat_lower), 2),
            "upper_bound": round(float(row.yhat_upper), 2),
        }
        for row in out.itertuples()
    ]


# --------------------------------------------------------------------------
# Training entry point
# --------------------------------------------------------------------------


def apply_train_window(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Keep only the most recent ``window_days`` rows (0 / None = keep all).

    This is the single highest-impact knob in the whole model -- see the
    backtest table in src/config.py.
    """
    if not window_days or window_days <= 0 or window_days >= len(df):
        return df.reset_index(drop=True)
    return df.iloc[-window_days:].reset_index(drop=True)


def train(
    features_path: Path | None = None,
    horizon: int = config.FORECAST_DAYS,
    mape_threshold: float = config.MAPE_THRESHOLD,
    train_window_days: int = config.TRAIN_WINDOW_DAYS,
    regressors: tuple[str, ...] = (),
    force_register: bool = False,
    **model_kwargs,
) -> dict:
    """Run the full validate-then-refit protocol and persist artifacts.

    Returns the metrics payload, including a ``registered`` flag telling the
    Airflow task whether the model cleared the threshold gate.
    """
    config.ensure_dirs()
    features_path = Path(features_path) if features_path else config.FEATURES_CSV

    if not features_path.exists():
        raise FileNotFoundError(
            f"feature file not found at {features_path}; run src.features first"
        )

    df = pd.read_csv(features_path, parse_dates=["ds"])
    if len(df) < horizon * 10:
        raise ValueError(f"only {len(df)} rows available; need at least {horizon * 10} to train")

    # --- 1. Hold-out split -------------------------------------------------
    # The test window is always the true final `horizon` days. The training
    # window is then truncated to the most recent TRAIN_WINDOW_DAYS *before*
    # that -- never overlapping the test window.
    train_df = apply_train_window(df.iloc[:-horizon], train_window_days)
    test_df = df.iloc[-horizon:].reset_index(drop=True)
    log.info(
        "Split: %d train rows (%s -> %s), %d test rows (%s -> %s)",
        len(train_df),
        train_df["ds"].min().date(),
        train_df["ds"].max().date(),
        len(test_df),
        test_df["ds"].min().date(),
        test_df["ds"].max().date(),
    )

    # --- 2. Validation fit -------------------------------------------------
    log.info("Fitting validation model on %d rows...", len(train_df))
    _, val_forecast = fit_and_forecast(train_df, horizon, regressors, **model_kwargs)

    metrics = compute_metrics(test_df["y"].to_numpy(), val_forecast["yhat"].to_numpy())
    log.info(
        "Hold-out metrics: MAE=%.2f RMSE=%.2f MAPE=%.3f%%",
        metrics["mae"],
        metrics["rmse"],
        metrics["mape"],
    )

    passed = metrics["mape"] <= mape_threshold
    registered = passed or force_register

    # --- 3. Refit on full history and persist ------------------------------
    # Millisecond resolution, not seconds: a retrained model must always get a
    # distinct version. At second resolution an Airflow retry (or two runs in
    # the same second) produced a colliding version, which made /reload report
    # changed=false for a model that had genuinely changed, and clobbered the
    # previous run.json in RUNS_DIR.
    model_version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    payload = {
        "model_version": model_version,
        "model_type": "prophet",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon_days": horizon,
        "train_rows": len(train_df),
        "train_start": str(train_df["ds"].min().date()),
        "train_end": str(train_df["ds"].max().date()),
        "test_start": str(test_df["ds"].min().date()),
        "test_end": str(test_df["ds"].max().date()),
        "train_window_days": train_window_days or len(df),
        "regressors": list(regressors),
        "mape_threshold": mape_threshold,
        "threshold_passed": passed,
        "registered": registered,
        **metrics,
    }

    if registered:
        if not passed:
            log.warning(
                "MAPE %.3f%% exceeds threshold %.2f%% but --force-register was set",
                metrics["mape"],
                mape_threshold,
            )
        # Refit including the held-out week so the live forecast starts from
        # today rather than from a week ago, under the same window policy.
        full_df = apply_train_window(df, train_window_days)
        log.info(
            "Refitting on %d rows (%s -> %s) for the production artifact...",
            len(full_df),
            full_df["ds"].min().date(),
            full_df["ds"].max().date(),
        )
        full_model, full_forecast = fit_and_forecast(full_df, horizon, regressors, **model_kwargs)

        save_model(full_model, config.MODEL_JSON)
        payload["forecast"] = forecast_to_records(full_forecast)
        save_json(
            {"model_version": model_version, "forecast": payload["forecast"]}, config.FORECAST_JSON
        )
        save_json(payload, config.METRICS_JSON)
        log.info("Registered model version %s", model_version)
    else:
        log.warning(
            "REJECTED: MAPE %.3f%% exceeds threshold %.2f%%. "
            "Existing model artifact left untouched.",
            metrics["mape"],
            mape_threshold,
        )

    # --- 4. Local run tracking (no MLflow server; lives on the PVC) --------
    run_dir = config.RUNS_DIR / model_version
    save_json(payload, run_dir / "run.json")

    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the 7-day BTC Prophet forecaster.")
    parser.add_argument("--features", default=None)
    parser.add_argument("--horizon", type=int, default=config.FORECAST_DAYS)
    parser.add_argument("--mape-threshold", type=float, default=config.MAPE_THRESHOLD)
    parser.add_argument(
        "--train-window-days",
        type=int,
        default=config.TRAIN_WINDOW_DAYS,
        help="train on the most recent N days only; 0 = all history",
    )
    parser.add_argument(
        "--changepoint-prior-scale", type=float, default=config.CHANGEPOINT_PRIOR_SCALE
    )
    parser.add_argument("--changepoint-range", type=float, default=config.CHANGEPOINT_RANGE)
    parser.add_argument(
        "--seasonality-mode",
        default=config.SEASONALITY_MODE,
        choices=["additive", "multiplicative"],
    )
    parser.add_argument(
        "--regressors",
        default="",
        help="comma-separated subset of "
        f"{','.join(CANDIDATE_REGRESSORS)} (off by default; see docstring)",
    )
    parser.add_argument(
        "--force-register",
        action="store_true",
        help="persist the model even if it fails the MAPE gate",
    )
    parser.add_argument(
        "--build-features", action="store_true", help="rebuild features from raw data first"
    )
    args = parser.parse_args(argv)

    if args.build_features:
        features.run()

    regressors = tuple(r.strip() for r in args.regressors.split(",") if r.strip())
    bad = [r for r in regressors if r not in CANDIDATE_REGRESSORS]
    if bad:
        parser.error(f"unknown regressors {bad}; choose from {list(CANDIDATE_REGRESSORS)}")

    payload = train(
        features_path=args.features,
        horizon=args.horizon,
        mape_threshold=args.mape_threshold,
        train_window_days=args.train_window_days,
        regressors=regressors,
        force_register=args.force_register,
        changepoint_prior_scale=args.changepoint_prior_scale,
        changepoint_range=args.changepoint_range,
        seasonality_mode=args.seasonality_mode,
    )

    print("\n" + "=" * 68)
    print(f"  MODEL VERSION : {payload['model_version']}")
    print(
        f"  TRAIN WINDOW  : {payload['train_start']} -> {payload['train_end']} "
        f"({payload['train_rows']} rows)"
    )
    print(
        f"  TEST WINDOW   : {payload['test_start']} -> {payload['test_end']} "
        f"({payload['horizon_days']} days held out)"
    )
    print("-" * 68)
    print(f"  MAE           : {payload['mae']:,.2f} USD")
    print(f"  RMSE          : {payload['rmse']:,.2f} USD")
    print(f"  MAPE          : {payload['mape']:.3f} %")
    print(
        f"  THRESHOLD     : {payload['mape_threshold']:.2f} % "
        f"-> {'PASS' if payload['threshold_passed'] else 'FAIL'}"
    )
    print(f"  REGISTERED    : {payload['registered']}")
    print("=" * 68)

    if payload.get("forecast"):
        print("\n  7-DAY FORECAST (refit on full history)\n")
        print(f"  {'DATE':<12} {'PREDICTED':>12} {'LOWER 95%':>12} {'UPPER 95%':>12}")
        print("  " + "-" * 50)
        for row in payload["forecast"]:
            print(
                f"  {row['date']:<12} {row['predicted_close']:>12,.2f} "
                f"{row['lower_bound']:>12,.2f} {row['upper_bound']:>12,.2f}"
            )
        print()

    return 0 if payload["registered"] else 2


if __name__ == "__main__":
    sys.exit(main())
