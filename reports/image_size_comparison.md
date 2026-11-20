# Docker image size: Prophet optimisation

Measured on the Part B build, `app/Dockerfile`, runtime base `python:3.10-slim`.

| Build | Size | Delta |
|---|---|---|
| Unoptimised (single-stage deps, scikit-learn present, nothing stripped) | **913 MB** | baseline |
| Optimised (multi-stage, no scikit-learn, matplotlib stripped) | **513 MB** | **-400 MB (-44%)** |
| `python:3.10-slim` base alone | ~125 MB | floor |

## Where the 400 MB came from

| Change | Saved | Reasoning |
|---|---|---|
| Dropped `scikit-learn` | ~167 MB | sklearn (56 MB) + its scipy dependency (111 MB). Never imported: MAE/RMSE/MAPE in `src/train_forecast.py` are plain numpy, and Prophet does not require scipy. |
| Stripped `matplotlib` from runtime | ~34 MB | Prophet hard-requires it, but `prophet/plot.py` guards the import in `try/except ImportError`. Inference and fitting both work without it; only plotting is lost, which the API never does. |
| Multi-stage build | ~200 MB | `build-essential`, pip caches and `__pycache__` stay in the builder stage; only the finished venv is copied into the runtime. |

## What cannot be removed

`prophet/stan_model` is 45 MB of compiled Stan binary and is required — the
`/predict` custom-history path fits a new model at request time, not just
deserialising a stored one. This is the irreducible cost of choosing Prophet.

## Why Prophet over NeuralProphet

NeuralProphet pulls in PyTorch, which is ~800 MB on its own and would roughly
triple the final image, for a model whose accuracy advantage on a 7-day daily
horizon is not established. Prophet's 513 MB was accepted as a deliberate
tradeoff: a moderately large image in exchange for a dependency-light,
CPU-only, fast-fitting model that trains in ~0.2 s and serialises to a 60 KB
JSON artifact the Flask pods can reload without a restart.

A build guard enforces the matplotlib strip: `app/Dockerfile` runs an import
smoke test that fails the build if stripping matplotlib ever breaks Prophet,
rather than shipping an image that dies on first request.
