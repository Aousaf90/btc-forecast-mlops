"""Daily BTC-USD ETL, retraining and model-reload DAG (Part C1).

Four tasks, strictly ordered:

    fetch_btc_data -> preprocess_features -> retrain_and_evaluate -> trigger_model_reload

The pipeline modules live in ``src/`` and are imported here rather than
reimplemented, so the DAG, the CLI and the test suite all exercise the same
code path. docker-compose mounts ./src at /opt/airflow/project/src and puts it
on PYTHONPATH.

Two design notes:

* **The reload task uses ``requests`` inside a PythonOperator, not
  SimpleHttpOperator.** SimpleHttpOperator requires an Airflow HTTP Connection
  to be configured in the UI before it will run -- an extra manual step and a
  silent failure point when demonstrating the DAG on a fresh database. A plain
  request against PREDICTION_API_URL needs no such setup and gives a clearer
  error when the API is down.

* **The threshold gate lives in the training task**, which registers the model
  only if its hold-out MAPE passes. The reload task reads that decision from
  XCom and skips itself when nothing new was registered -- reloading an
  unchanged artifact would be a no-op that falsely signals a fresh deploy.

* **The pipeline modules are imported INSIDE each task, never at module
  level.** Airflow re-parses every DAG file on a short interval (default 30s),
  and this file's dependencies pull in Prophet, pandas and numpy. Importing
  them at parse time stalled the scheduler -- it logged
  "WORKER TIMEOUT" and "Heartbeat recovered after 559 seconds", and never
  serialized the DAG into DagModel, so `airflow dags trigger` failed with
  DagNotFound even though `airflow dags list` could see the file. Deferring
  the imports into the callables keeps parsing to a few milliseconds; the cost
  is paid once per task execution instead of every 30 seconds.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
from airflow.exceptions import AirflowSkipException
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

# The pipeline package is mounted at /opt/airflow/project. Add it defensively
# so the DAG still resolves if PYTHONPATH is not set in some other environment.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
if PROJECT_ROOT.is_dir() and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: src.* and requests are imported inside the task callables, not here.
# See the module docstring -- top-level Prophet imports stall the scheduler.

LOCAL_TZ = pendulum.timezone("UTC")

PREDICTION_API_URL = os.getenv("PREDICTION_API_URL", "http://api:5000")
RELOAD_TIMEOUT_SECONDS = 30

DEFAULT_ARGS = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 3,
    # yfinance rate-limits and the providers back off internally; spacing Airflow
    # retries further apart avoids stacking a second burst on top of that.
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def fetch_btc_data(**context) -> str:
    """Task 1: pull daily BTC-USD OHLCV into shared storage.

    Tries yfinance, then Binance, then CoinGecko, retrying each with
    exponential backoff before failing over. Falls back to the cached raw CSV
    if every provider is unavailable, so a rate-limited day degrades instead
    of breaking the pipeline.
    """
    import pandas as pd

    from src import config, etl_fetch

    path = etl_fetch.run(period=os.getenv("HISTORY_PERIOD", config.HISTORY_PERIOD))
    df = pd.read_csv(path)
    context["ti"].xcom_push(key="raw_rows", value=len(df))
    context["ti"].xcom_push(key="latest_date", value=str(df["Date"].max()))
    print(f"Fetched {len(df)} rows into {path} (latest: {df['Date'].max()})")
    return str(path)


def preprocess_features(**context) -> str:
    """Task 2: clean the series and engineer the technical indicators.

    Produces 7- and 30-day moving averages and annualised rolling volatility,
    output with Prophet's required ds/y column names.

    The indicators are persisted but deliberately NOT passed to Prophet's
    add_regressor: a regressor needs a value for every future date being
    predicted, and a 7-day moving average over the next 7 days depends on the
    very prices being forecast. See src/features.py for the full reasoning.
    """
    import pandas as pd

    from src import features

    path = features.run()
    df = pd.read_csv(path)
    context["ti"].xcom_push(key="feature_rows", value=len(df))
    context["ti"].xcom_push(key="feature_columns", value=list(df.columns))
    print(f"Built {len(df)} feature rows -> {path}")
    print(f"Columns: {list(df.columns)}")
    return str(path)


def retrain_and_evaluate(**context) -> dict:
    """Task 3: fit, score on the latest 7-day window, register if it passes.

    Holds out the final 7 days, scores MAE/RMSE/MAPE against them, and only if
    the MAPE clears MAPE_THRESHOLD refits on the full training window and
    writes the artifact. A model that fails the gate leaves the currently
    deployed artifact untouched.
    """
    from src import config, train_forecast

    payload = train_forecast.train(
        horizon=config.FORECAST_DAYS,
        mape_threshold=config.MAPE_THRESHOLD,
        train_window_days=config.TRAIN_WINDOW_DAYS,
    )

    ti = context["ti"]
    ti.xcom_push(key="registered", value=bool(payload["registered"]))
    ti.xcom_push(key="model_version", value=payload["model_version"])
    ti.xcom_push(key="mae", value=payload["mae"])
    ti.xcom_push(key="rmse", value=payload["rmse"])
    ti.xcom_push(key="mape", value=payload["mape"])

    print(
        f"Evaluation on {payload['test_start']}..{payload['test_end']}: "
        f"MAE={payload['mae']:,.2f}  RMSE={payload['rmse']:,.2f}  "
        f"MAPE={payload['mape']:.3f}%"
    )
    if payload["registered"]:
        print(f"REGISTERED model version {payload['model_version']}")
    else:
        print(
            f"REJECTED: MAPE {payload['mape']:.3f}% exceeds threshold "
            f"{payload['mape_threshold']:.2f}%. Live artifact left unchanged."
        )

    # XComs are JSON-serialised; return only plain types.
    return {
        "model_version": payload["model_version"],
        "registered": payload["registered"],
        "mae": payload["mae"],
        "rmse": payload["rmse"],
        "mape": payload["mape"],
    }


def trigger_model_reload(**context) -> dict:
    """Task 4: tell the Flask service to re-read the model from disk.

    Skips itself when task 3 registered nothing -- there is no new artifact to
    load, and a successful reload of an unchanged model would misleadingly
    suggest a fresh deployment.
    """
    import requests

    ti = context["ti"]
    registered = ti.xcom_pull(task_ids="retrain_and_evaluate", key="registered")
    model_version = ti.xcom_pull(task_ids="retrain_and_evaluate", key="model_version")

    if not registered:
        raise AirflowSkipException(
            "no new model was registered (threshold not met); nothing to reload"
        )

    url = f"{PREDICTION_API_URL.rstrip('/')}/reload"
    print(f"POST {url}  (expecting model_version {model_version})")

    try:
        response = requests.post(url, timeout=RELOAD_TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise RuntimeError(f"could not reach the prediction API at {url}: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"reload failed with HTTP {response.status_code}: {response.text[:300]}")

    body = response.json()
    print(
        f"Reloaded: previous={body.get('previous_version')} "
        f"-> current={body.get('model_version')} (changed={body.get('changed')})"
    )

    if body.get("model_version") != model_version:
        # Not fatal: with multiple API replicas behind a Service, this request
        # reaches one pod. Surfaced as a warning so it is visible in the logs.
        print(
            f"WARNING: API reports {body.get('model_version')} but this run "
            f"registered {model_version}. With several replicas, only the pod "
            f"that served this request has reloaded."
        )

    return body


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="btc_etl_pipeline",
    description="Daily BTC-USD ingestion, feature engineering, retraining and API reload",
    default_args=DEFAULT_ARGS,
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=LOCAL_TZ),
    # No backfilling: the pipeline always trains on the latest available
    # history, so replaying past intervals would repeat identical work.
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "bitcoin", "prophet", "forecasting"],
    doc_md=__doc__,
) as dag:

    task_fetch = PythonOperator(
        task_id="fetch_btc_data",
        python_callable=fetch_btc_data,
        doc_md="Pull daily BTC-USD OHLCV from yfinance/Binance/CoinGecko into shared storage.",
    )

    task_preprocess = PythonOperator(
        task_id="preprocess_features",
        python_callable=preprocess_features,
        doc_md="Clean the series; compute 7/30-day moving averages and volatility in ds/y format.",
    )

    task_retrain = PythonOperator(
        task_id="retrain_and_evaluate",
        python_callable=retrain_and_evaluate,
        doc_md="Fit Prophet, score the latest 7-day window, register only if MAPE passes.",
    )

    task_reload = PythonOperator(
        task_id="trigger_model_reload",
        python_callable=trigger_model_reload,
        doc_md="POST /reload to the Flask service so running pods pick up the new artifact.",
    )

    task_fetch >> task_preprocess >> task_retrain >> task_reload
