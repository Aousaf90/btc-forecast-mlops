# Project Handoff — BTC 7-Day Forecasting MLOps Pipeline

**Purpose of this file.** Hand this to a new Claude Code session (or a teammate) on
a different machine so work can continue without re-deriving context. It records
what exists, what has actually been *verified* versus merely written, the decisions
that are locked, the traps already hit, and exactly what comes next.

**Last updated:** 2026-11-21, after Part C.
**Repository:** https://github.com/Aousaf90/btc-forecast-mlops
**Docker image:** `aousafs/btc-forecast-api` (tags: `latest`, `sha-<commit>`)

> Read `README.md` first for the architecture, project structure and API reference.
> This file covers state and intent, not documentation.

---

## 1. The assignment

10-day group-of-2 Master's assignment, 60 marks. Build and deploy a production MLOps
system: BTC-USD forecasting model → Flask REST API → Docker → GitHub Actions CI/CD →
Airflow + Kubeflow orchestration → Kubernetes with autoscaling → PDF report.

The spec PDF is committed at the repo root:
`Assignment 3_Flask_Docker_Airflow_Kubeflow_K8s.pdf`. **Read it before continuing** —
the mark allocations and exact numeric requirements are graded literally.

| Part | Scope | Marks | Status |
|---|---|---|---|
| A | Flask REST API + pytest | 10 | ✅ Complete, verified |
| B | Docker + CI/CD | 10 | ✅ Complete, verified |
| C | Airflow DAG + Kubeflow | 12 | ✅ Complete, verified |
| D | Kubernetes deployment | 18 | ⬜ **Next** |
| E | Verification + 4–6 page PDF report | 10 | ⬜ Not started |

---

## 2. Locked decisions — do not substitute these

These were decided deliberately. Changing one means redoing work.

| Decision | Rationale |
|---|---|
| **Prophet, not NeuralProphet** | NeuralProphet pulls PyTorch (~800 MB), roughly tripling the image, for no established accuracy gain at a 7-day daily horizon |
| **`model_to_json` / `model_from_json`, never pickle** | A fitted Prophet does not round-trip reliably across versions, and the Airflow container writes the artifact the Flask container reads |
| **Identical Prophet pin in both images** | A drift breaks `/reload` at runtime, not build time |
| **No MLflow server** | Run tracking writes to `models/runs/<version>/run.json` on the PVC — one less service to deploy |
| **Model artifacts and data never committed** | They live on the PVC; `.gitignore` enforces this |
| **Moving averages NOT fed to `add_regressor`** | See §5 |
| **Kubeflow compiled only, never installed** | A full Kubeflow install needs ~16 GB RAM; the rubric only requires the YAML |
| **All commits authored as Aousaf90** | Submission ownership. `git config --local` is already pinned in the repo |

---

## 3. What is actually verified (not just written)

Everything below was executed and observed, not assumed.

**Part A**
- 85 pytest tests pass (46 API, 39 pipeline)
- All four endpoints served live under gunicorn
- Prophet JSON round-trip confirmed: forecast after reload matched pre-save to **$0.000000**

**Part B**
- Image builds at **513 MB** (down from 913 MB unoptimised — see `reports/image_size_comparison.md`)
- Runs as non-root `appuser` (uid 1000)
- Compose stack: all 4 services healthy
- **Full handoff proven in compose**: trained in the Airflow container → API container
  saw the artifact on the shared volume → `POST /reload` returned `changed: true` →
  new `model_version` served by `/predict`
- GitHub Actions green: lint 13s, 85 tests 46s, build+push 39s (layer cache working)
- Docker Hub has `latest` + full-SHA + short-SHA tags

**Part C**
- DAG registered, no import errors, visible in the Airflow UI
- Manual run: **all 4 tasks succeeded**
  - `fetch_btc_data` → 1,824 rows
  - `preprocess_features` → 1,794 rows incl. `ma_7, ma_30, volatility_7, volatility_30`
  - `retrain_and_evaluate` → MAE 6,162.80 / RMSE 6,272.82 / MAPE 9.583% → REGISTERED
  - `trigger_model_reload` → `previous=…171256183 → current=…171404345, changed=True`
- KFP spec compiles: 3 container components, correct dependency chain
- Each KFP component's command verified to exist in the published image

---

## 4. Traps already hit — do not reintroduce

Each of these cost real debugging time. They are documented in-code at the site of
the fix; this is the index.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | yfinance returns nothing, no exception | Yahoo answers **HTTP 429**; yfinance surfaces it as an *empty DataFrame* | Treat empty as failure; fail over to Binance → CoinGecko (`src/providers.py`) |
| 2 | MAPE 18.43%, model useless | 5 years of history + Prophet's `changepoint_range=0.8` left no changepoints in the recent window | 365-day training window, `changepoint_range=0.95` → **3.49%** |
| 3 | `/reload` reported `changed: false` for a genuinely new model | `model_version` used **second** resolution; two trainings in one second collided | Millisecond resolution (`src/train_forecast.py`) |
| 4 | Airflow scheduler crash-loop: `ImportError: cannot import name '_app_ctx_stack' from 'flask'` | Shared `requirements.txt` pinned Flask 3.0.3, clobbering the Flask 2.2.x Airflow's `flask-appbuilder` needs | Split `requirements.txt` (shared core) / `requirements-api.txt` (Flask+gunicorn). Build guard in `airflow/Dockerfile` |
| 5 | `PermissionError: /opt/airflow/data/raw` | Docker creates named volumes root-owned; Airflow runs as uid 50000 | `airflow-init` runs as root solely to chown them |
| 6 | DAG visible to `airflow dags list` but `dags trigger` → `DagNotFound` | `from src import …` at DAG module level pulled Prophet into every ~30s parse cycle; scheduler timed out (`Heartbeat recovered after 559 seconds`) and never serialized to `DagModel` | **All heavy imports inside task callables.** Parse time 0.135s |
| 7 | KFP: `Artifacts must have both a schema_title and a schema_version` | `from __future__ import annotations` (PEP 563) stringifies annotations; the KFP compiler needs real objects | Omit that import in `kubeflow/btc_kfp_pipeline.py` |
| 8 | `.gitignore` silently swallowed graded evidence | Unanchored `logs/` rule also matched `screenshots/logs/` | Root-scoped `/logs/` |
| 9 | Shell dies mid-script, exit 144 | `pkill -f gunicorn` matched the wrapper shell's own command line | Kill by captured PID |

---

## 5. The regressor decision (asked about repeatedly — settled)

The rubric requires the preprocessing task to compute moving averages and
volatility. It does, and they are persisted to `data/processed/btc_features.csv`.

They are **not** passed to Prophet's `add_regressor`, deliberately. A Prophet
regressor needs a value for every *future* date being predicted. A 7-day moving
average over the next 7 days depends on the very prices being forecast — supplying
it means extrapolating a guess and leaking that guess into the model.

An opt-in path exists (`--regressors ma_7,…`, flat carry-forward) but is off by
default. **This is a named, reasoned tradeoff and belongs in the report.**

---

## 6. Environment notes

**Development machine (current):** Linux, 7.6 GB RAM, 4 GB swap, 8 cores, Docker.
Python 3.12 locally; containers run Python 3.10.

**Memory budget — measured, not estimated:**
- gunicorn worker with model loaded: **116 MB**; master + 2 workers ≈ **258 MB**
- Comfortably inside the graded 512Mi pod limit
- The graded 128Mi *request* being below actual usage is fine — Kubernetes OOMKills
  on **limits**, never on requests

**⚠️ Never run `docker compose` and Minikube simultaneously** — together they exceed
7.6 GB. `docker compose down` before `minikube start`.

**Accounts:** `gh` CLI and Docker CLI are both authenticated as Aousaf on the current
machine. On a **new machine** these must be set up again:
```bash
gh auth login                                    # as Aousaf90
docker login -u aousafs                          # use a PAT, not the password
git config --local user.name  "Aousaf90"
git config --local user.email "aousafsuleman@gmail.com"
```
GitHub Secrets `DOCKER_USERNAME` / `DOCKER_PASSWORD` are already set on the repo.
**Never put credentials in files, commits, or chat.**

---

## 7. NEXT: Part D — Kubernetes (18 marks, the heaviest section)

### Setup (run first, on the target machine)

```bash
docker compose down                       # ⚠️ required

minikube start --driver=docker --memory=3600 --cpus=4
minikube addons enable metrics-server     # required for the HPA
minikube addons enable ingress            # required for the Ingress
echo "$(minikube ip) btc-forecast.local" | sudo tee -a /etc/hosts
kubectl get nodes
```

### Manifests to write in `k8s/` — these numbers are graded literally

| File | Requirement |
|---|---|
| `configmap-secret.yaml` | ConfigMap: `MODEL_STAGE`, `LOG_LEVEL`, `PORT`. Secret: API keys/credentials (**dummy base64 only**). Both mounted as env vars |
| `pvc.yaml` | PVC retaining model artifacts and data across restarts, mounted at **`/app/models`** |
| `deployment.yaml` | **3 replicas**; liveness `GET /health` `initialDelaySeconds: 15`, `periodSeconds: 10`; readiness `GET /health`; requests **cpu 100m / memory 128Mi**; limits **cpu 500m / memory 512Mi** |
| `service.yaml` | ClusterIP **and** NodePort, **port 80 → targetPort 5000** |
| `hpa.yaml` | **minReplicas 2, maxReplicas 8, CPU target 60%** |
| `ingress.yaml` | NGINX, host **`btc-forecast.local`** → the service |

> ⚠️ The rubric says **3 replicas** in the Deployment but **minReplicas 2** in the
> HPA. That inconsistency is in the spec as written. **Implement both exactly as
> stated — do not "fix" it.** Mention it in the report instead.

> ⚠️ If pods get OOMKilled at 512Mi, **stop and ask the user** rather than raising
> the limit. The numbers are graded. (Measured usage says this should not happen.)

### Then
- Load test with `wrk` or Locust; capture `kubectl get hpa --watch` scaling 2→8
- **Part E:** delete a pod and document self-healing; re-run the DAG end-to-end
  against the cluster; query the live endpoint; write the 4–6 page PDF report

---

## 8. Evidence collection — Part E is 10 marks and cannot be reconstructed

`screenshots/logs/` holds committed text transcripts. `scripts/capture_part_a.sh`
regenerates Part A evidence and scrubs `$HOME` from paths automatically.

**Screenshots still needed:** Parts A–C captures (the CLI transcripts exist, but the
report needs images of the Airflow DAG graph and the GitHub Actions run), then all of
Part D and E.

Report must cover A–E with: Docker build output, Airflow DAG graph, GitHub Actions
runs, `kubectl get pods/svc/hpa/ingress`, load-test autoscaling output, sample
predictions, an architecture diagram, and a section justifying the **Prophet choice
and its image-size tradeoff** (913 MB → 513 MB; see `reports/image_size_comparison.md`).

---

## 9. How to work on this project

Conventions the user has asked for and that the existing code follows:

1. **Each phase must actually run before moving on.** Do not write Part D and
   declare it done — apply the manifests and observe the pods.
2. **Flag what needs a screenshot**, explicitly, when it happens.
3. **Give exact commands** to run.
4. **Stop and ask** before anything touching the user's accounts (Docker Hub repos,
   GitHub repo settings, GitHub Secrets).
5. **Never commit credentials.** The Secret manifest uses dummy base64 values.
6. **Meaningful commit messages at every stage**, explaining *why*, not just what.
7. **Report honestly.** If something fails, say so with the output.

### Verify the project still works after cloning

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest tests/ -q                       # expect 85 passed
./venv/bin/python -m src.etl_fetch && ./venv/bin/python -m src.features
./venv/bin/python -m src.train_forecast                     # expect MAPE well under 15%
./venv/bin/python kubeflow/btc_kfp_pipeline.py              # expect a compiled YAML
```
