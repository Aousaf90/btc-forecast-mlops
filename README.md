# BTC 7-Day Forecasting — Production MLOps Pipeline

End-to-end MLOps system that ingests daily Bitcoin (BTC-USD) price data, trains a
Prophet time-series model, and serves a 7-day closing-price forecast over a REST
API — containerised, CI/CD'd through GitHub Actions, orchestrated with Airflow and
Kubeflow, and deployed on Kubernetes with autoscaling.

| | |
|---|---|
| **Repository** | https://github.com/Aousaf90/btc-forecast-mlops |
| **Docker image** | [`aousafs/btc-forecast-api`](https://hub.docker.com/r/aousafs/btc-forecast-api) |
| **Model** | Facebook Prophet 1.1.6 |
| **Forecast horizon** | 7 days |
| **Current hold-out accuracy** | MAE $6,162.80 · RMSE $6,272.82 · **MAPE 9.58%** |
| **Rolling backtest (12 windows)** | mean MAPE **3.49%** (best 0.92%, worst 9.58%) |

---

## 1. Architecture

```
                   ┌─────────────────────────────────────────────┐
                   │  Market data: yfinance → Binance → CoinGecko │
                   └───────────────────────┬─────────────────────┘
                                           │ (retry + failover)
┌──────────────────────────────────────────▼──────────────────────────────────┐
│  Airflow DAG  btc_etl_pipeline  (@daily)                                     │
│                                                                              │
│   fetch_btc_data → preprocess_features → retrain_and_evaluate → trigger_…    │
│        │                  │                      │                  │        │
│     raw CSV          ds/y + MA/vol        MAPE gate (≤15%)     POST /reload  │
└────────┴──────────────────┴──────────────────────┴──────────────────┼────────┘
                                           │                          │
                            writes         ▼                          │
                   ┌───────────────────────────────────┐              │
                   │  Shared storage (PVC / volume)    │              │
                   │  prophet_model.json, metrics.json │              │
                   └───────────────────┬───────────────┘              │
                              reads    │                              │
┌────────────────────────────────────── ▼──────────────────────────┐  │
│  Flask API (gunicorn)   /health  /predict  /metrics  /reload  ◄───┼──┘
│  Kubernetes: 3 replicas, HPA 2→8 @60% CPU, NGINX Ingress         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Project structure

| Path | Purpose |
|---|---|
| `app/app.py` | Flask REST API — `/health`, `/predict`, `/metrics`, `/reload` |
| `app/Dockerfile` | Multi-stage production image (513 MB, non-root) |
| `src/config.py` | Shared paths and hyperparameters, all env-overridable |
| `src/providers.py` | Market-data providers with failover (yfinance → Binance → CoinGecko) |
| `src/etl_fetch.py` | Daily OHLCV ingestion: retry, backoff, raw CSV cache |
| `src/features.py` | Cleaning + 7/30-day moving averages and volatility, in Prophet `ds`/`y` format |
| `src/train_forecast.py` | Train, evaluate on a held-out week, register if it passes the gate |
| `src/backtest.py` | Rolling-origin backtest used to choose the hyperparameters |
| `dags/btc_etl_pipeline.py` | Airflow DAG — 4 tasks, daily schedule |
| `kubeflow/btc_kfp_pipeline.py` | KFP pipeline (3 container components) |
| `kubeflow/btc_kfp_pipeline.yaml` | Compiled pipeline spec |
| `.github/workflows/ci_cd.yml` | Lint → test → build → push to Docker Hub |
| `airflow/Dockerfile` | Airflow worker image (same pinned Prophet as the API) |
| `docker-compose.yml` | Local stack: Flask + Airflow + Postgres |
| `tests/` | 85 pytest tests (46 API, 39 pipeline) |
| `k8s/` | Kubernetes manifests — **not yet written (Part D)** |

**Not in git** (by design): `models/`, `data/`, `venv/`, `logs/`. Model artifacts and
market data live on the Kubernetes PVC, never in the repository.

---

## 3. Quick start

### Prerequisites
Docker, Python 3.10+, and (for Part D) Minikube, kubectl, Helm.

### Local setup

```bash
git clone https://github.com/Aousaf90/btc-forecast-mlops.git
cd btc-forecast-mlops

python3 -m venv venv
./venv/bin/pip install -r requirements-dev.txt
```

### Train a model and see a forecast

```bash
./venv/bin/python -m src.etl_fetch          # fetch BTC-USD into data/raw/
./venv/bin/python -m src.features           # build data/processed/btc_features.csv
./venv/bin/python -m src.train_forecast     # train, evaluate, register
```

Prints MAE/RMSE/MAPE on a held-out 7-day window and the resulting forecast.

### Run the API

```bash
MODEL_DIR="$PWD/models" DATA_DIR="$PWD/data" \
  ./venv/bin/gunicorn --chdir app --bind 0.0.0.0:5000 --workers 2 app:app
```

```bash
curl -s localhost:5000/health  | python -m json.tool
curl -s localhost:5000/metrics | python -m json.tool
curl -s -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{}'
curl -s -X POST localhost:5000/predict -H 'Content-Type: application/json' \
     -d '{"forecast_days": 3}'
```

### Run the tests

```bash
./venv/bin/python -m pytest tests/ -v
./venv/bin/python -m flake8 src/ app/ tests/ dags/ kubeflow/
./venv/bin/python -m black --check src/ app/ tests/ dags/ kubeflow/
```

### Full local stack (Flask + Airflow + Postgres)

```bash
docker compose up -d --build
```

| Service | URL | Credentials |
|---|---|---|
| Flask API | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | `airflow` / `airflow` |

Trigger the pipeline:

```bash
docker compose exec airflow-scheduler airflow dags unpause btc_etl_pipeline
docker compose exec airflow-scheduler airflow dags trigger btc_etl_pipeline
docker compose down          # ⚠️ required before starting Minikube — see §6
```

### Compile the Kubeflow pipeline

```bash
./venv/bin/python kubeflow/btc_kfp_pipeline.py
```

---

## 4. API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Status, uptime, loaded model version. Returns **200 whenever the process is serving**, with `model_loaded: false` if no artifact exists |
| `POST` | `/predict` | 7-day forecast. Optional `forecast_days` (1–90, default 7) or `historical_data` to fit an ad-hoc model |
| `GET` | `/metrics` | MAE / RMSE / MAPE from the latest validation run |
| `POST` | `/reload` | Re-read the model artifact from disk (called by the Airflow DAG) |

**Why `/health` always returns 200:** both the Kubernetes liveness and readiness
probes point at it. Returning 503 for a missing model would make the liveness probe
restart-loop a healthy pod on a fresh PVC, before Airflow has ever trained one.
`/predict` is the endpoint that refuses with 503.

Error contract: `400` malformed/null payload, `404` unknown route, `405` wrong
method, `500` internal, `503` no model loaded. All responses are JSON, never HTML.

---

## 5. Model

Prophet with a **365-day training window**, `changepoint_prior_scale=0.05`,
`changepoint_range=0.95`, multiplicative seasonality. Chosen by rolling-origin
backtest (`python -m src.backtest --compare`), not by guesswork:

| Configuration | Mean MAPE |
|---|---|
| Full 5y history, `changepoint_range=0.8` (Prophet defaults) | 14.70% |
| **365-day window, `changepoint_range=0.95`** | **3.49%** |

The training-window length dominates every other knob: Prophet fits one global
piecewise-linear trend, so BTC's 2021–2023 regime actively biases a 7-day
extrapolation. Prophet's default `changepoint_range=0.8` also places *no*
changepoints in the most recent 20% of the series — exactly the stretch a 7-day
forecast extrapolates from.

**Registration gate:** a retrained model replaces the live artifact only if its
hold-out MAPE ≤ 15%. That threshold sits above the worst observed window (9.58%,
a week BTC fell ~$70k→$62k) but below the untuned baseline's 14.70% mean, so it
still catches a genuinely broken model.

**On the technical indicators:** 7/30-day moving averages and volatility are
computed and persisted, but deliberately **not** passed to Prophet's
`add_regressor`. A regressor needs a value for every *future* date being predicted,
and a 7-day moving average over the next 7 days depends on the very prices being
forecast. Feeding it an extrapolated guess would leak that guess into the model.

**Serialization** uses `prophet.serialize.model_to_json`, never pickle — a fitted
Prophet does not round-trip reliably across versions, and here the Airflow
container writes the artifact the Flask container reads.

---

## 6. Operational notes

**Prophet version parity is load-bearing.** The Airflow container *writes* the model
JSON that the Flask container *reads*. `requirements.txt` holds the shared modelling
stack (including the exact Prophet pin) and is installed by both images;
`requirements-api.txt` adds Flask and gunicorn for the API only. Airflow pins its own
Flask 2.2.x — installing the API's Flask 3.x into the Airflow image crash-loops the
scheduler with `ImportError: cannot import name '_app_ctx_stack'`.

**Memory.** A gunicorn worker with the model loaded measures ~116 MB; master + 2
workers ≈ 258 MB, comfortably inside the 512Mi pod limit.

**⚠️ Do not run `docker compose` and Minikube at the same time.** Together they
exceed the 7.6 GB development machine. Run `docker compose down` before
`minikube start`.

**Yahoo Finance rate-limits aggressively** (HTTP 429), and yfinance surfaces that as
an *empty DataFrame* rather than an exception. The provider chain treats "empty" as
failure and fails over to Binance, which supplies full daily OHLCV with no API key.

---

## 7. Status

| Part | Scope | Marks | Status |
|---|---|---|---|
| **A** | Flask REST API + pytest | 10 | ✅ Complete — 85 tests passing |
| **B** | Docker + CI/CD | 10 | ✅ Complete — image published, CI green |
| **C** | Airflow DAG + Kubeflow | 12 | ✅ Complete — DAG run verified, spec compiled |
| **D** | Kubernetes deployment | 18 | ⬜ Not started |
| **E** | Verification + report | 10 | ⬜ Not started |

See [`HANDOFF.md`](HANDOFF.md) for detailed state, decisions and next steps.

---

## 8. Configuration

Every setting is an environment variable with a sensible default (`src/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Raw and processed CSVs |
| `MODEL_DIR` | `models` | Model artifacts (the PVC mount in Kubernetes) |
| `HISTORY_PERIOD` | `5y` | How much history to fetch |
| `FORECAST_DAYS` | `7` | Forecast horizon |
| `TRAIN_WINDOW_DAYS` | `365` | Train on the most recent N days (0 = all) |
| `MAPE_THRESHOLD` | `15.0` | Registration gate |
| `CHANGEPOINT_PRIOR_SCALE` | `0.05` | Prophet trend flexibility |
| `CHANGEPOINT_RANGE` | `0.95` | Fraction of history allowed changepoints |
| `MODEL_STAGE` | `production` | Reported by `/health` |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `PREDICTION_API_URL` | `http://api:5000` | Target for the DAG's reload task |

---

*Assignment 3 — Production MLOps Pipeline. Group submission.*
