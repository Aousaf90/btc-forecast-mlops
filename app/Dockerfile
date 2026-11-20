# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Multi-stage production image for the BTC forecasting Flask API.
#
# Prophet is a heavy dependency: its wheel bundles a ~45 MB compiled Stan
# binary, and it hard-requires matplotlib. Three things keep the final image
# as small as it can be while still running Prophet:
#
#   1. The whole dependency tree is installed into a virtualenv in a BUILDER
#      stage; only the finished venv is copied into the slim runtime. Build
#      toolchains (gcc, headers, pip's caches) never reach the final image.
#   2. scikit-learn/scipy are not dependencies at all (see requirements.txt):
#      that alone removes ~167 MB.
#   3. matplotlib is deleted from the runtime layer. prophet/plot.py guards
#      its matplotlib import in a try/except ImportError, so inference and
#      fitting both work without it -- only plotting, which the API never
#      does, is lost. Verified by the import smoke test at the end of this
#      stage, which fails the build if that ever stops being true.
# ---------------------------------------------------------------------------

# =========================== STAGE 1: builder ==============================
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed only if a wheel has to be compiled from source;
# it stays in this stage and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies are copied and installed BEFORE the application source, so that
# editing app code reuses this cached layer instead of reinstalling Prophet --
# which otherwise dominates every CI build.
COPY requirements.txt requirements-api.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && pip install -r /tmp/requirements-api.txt

# ---- Strip what inference never uses -------------------------------------
RUN set -eux; \
    SITE="$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf \
        "$SITE"/matplotlib "$SITE"/matplotlib-* "$SITE"/mpl_toolkits \
        "$SITE"/pylab.py "$SITE"/.libs_mpl* \
        "$SITE"/prophet/tests "$SITE"/pandas/tests "$SITE"/numpy/tests \
        "$SITE"/*.dist-info/RECORD.bak; \
    find "$SITE" -type d -name "__pycache__" -prune -exec rm -rf {} +; \
    find "$SITE" -type d -name "tests" -path "*/numpy/*" -prune -exec rm -rf {} +; \
    find /opt/venv -name "*.pyc" -delete

# Fail the build loudly if stripping matplotlib ever breaks Prophet, rather
# than shipping an image that dies on first request.
RUN /opt/venv/bin/python -c "\
import importlib.util; \
assert importlib.util.find_spec('matplotlib') is None, 'matplotlib was not stripped'; \
from prophet import Prophet; \
from prophet.serialize import model_to_json, model_from_json; \
import pandas, numpy, flask, gunicorn; \
print('builder smoke test OK: prophet imports without matplotlib')"

# =========================== STAGE 2: runtime ==============================
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="7-day Bitcoin close-price forecasting REST API (Prophet + Flask)" \
      org.opencontainers.image.source="https://github.com/Aousaf90/btc-forecast-mlops" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=5000 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_LEVEL=INFO \
    MODEL_STAGE=production

# curl is used by the container HEALTHCHECK below. --no-install-recommends and
# the apt list cleanup in the same layer keep this to a few MB.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (B1). UID/GID 1000 is fixed rather than auto-assigned so the
# PVC mounted at /app/models has a predictable owner in Kubernetes.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv

WORKDIR /app

# src/ before app/: src changes less often than the API module, so this orders
# the layers from most-stable to least-stable.
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser app/app.py /app/app.py

# /app/models is the PVC mount point (k8s/pvc.yaml); /app/data holds the raw
# and processed CSVs. Both are created owned by appuser so the non-root
# process can write to them even before a volume is attached.
RUN mkdir -p /app/models /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

# Mirrors the Kubernetes liveness probe so `docker run` and `docker compose`
# surface the same health signal locally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Production WSGI server (A3). Exec form so gunicorn is PID 1 and receives
# SIGTERM directly, giving Kubernetes a clean, fast pod shutdown.
#
# One worker, two threads: each worker holds its own deserialized Prophet model
# (~116 MB measured), so worker count is the main driver of pod memory against
# the 512Mi limit in k8s/deployment.yaml.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "--workers", "1", "--threads", "2", \
     "--timeout", "120", "--graceful-timeout", "30", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "app:app"]
