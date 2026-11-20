"""Flask REST API serving the 7-day BTC close-price forecast (Part A).

Endpoints
---------
GET  /health    liveness + readiness: status, uptime, loaded model version
POST /predict   7-day forecast; optional forecast_days or custom history
GET  /metrics   MAE / RMSE / MAPE from the latest validation run
POST /reload    re-read the model artifact from disk (called by the Airflow DAG)

Run in production with:
    gunicorn --bind 0.0.0.0:5000 app:app

A note on /health serving BOTH probes (k8s/deployment.yaml points the liveness
and readiness probes at it, as the rubric specifies): it returns 200 whenever
the *process* is healthy, even when no model artifact exists yet, and reports
``model_loaded: false`` in the body instead. Returning 503 for a missing model
would make the liveness probe restart-loop a perfectly healthy pod on a fresh
PersistentVolumeClaim -- before Airflow has ever had a chance to train one.
/predict is the endpoint that refuses (503) when no model is loaded.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# The container lays this out as /app/app.py beside /app/src/, so the repo root
# is already importable. Locally, gunicorn may be launched from inside app/ --
# bootstrap the parent onto sys.path so `from src import ...` resolves either way.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent):
    if (_candidate / "src").is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import pandas as pd  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

from src import config, train_forecast  # noqa: E402

log = config.get_logger("api")

START_TIME = time.time()

# Guard rails for the custom-history path on /predict.
MIN_HISTORY_POINTS = 30
MAX_HISTORY_POINTS = 5000
MAX_FORECAST_DAYS = 90


# ---------------------------------------------------------------------------
# Model store
# ---------------------------------------------------------------------------


class ModelStore:
    """Holds the deserialized Prophet model and its metadata.

    Guarded by a lock because /reload can fire while a /predict request is in
    flight. Each gunicorn worker keeps its own instance, so the Airflow DAG
    fans /reload out to the Service and every worker re-reads independently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model = None
        self._metrics: dict = {}
        self._version: str | None = None
        self._loaded_at: str | None = None

    def load(self) -> dict:
        """(Re-)read the model and metrics from disk. Returns a status dict."""
        with self._lock:
            model_path = config.MODEL_JSON
            if not model_path.exists():
                self._model = None
                self._version = None
                self._loaded_at = None
                log.warning("No model artifact at %s", model_path)
                return {"loaded": False, "reason": f"no model artifact at {model_path}"}

            model = train_forecast.load_model(model_path)

            metrics: dict = {}
            if config.METRICS_JSON.exists():
                import json

                try:
                    metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
                except ValueError as exc:
                    log.warning("metrics.json is not valid JSON (%s); ignoring", exc)

            self._model = model
            self._metrics = metrics
            # Fall back to the artifact mtime when metrics.json is absent, so a
            # model is never reported as version-less.
            self._version = metrics.get("model_version") or datetime.fromtimestamp(
                model_path.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y%m%d%H%M%S%f")[:-3]
            self._loaded_at = datetime.now(timezone.utc).isoformat()

            log.info("Loaded model version %s from %s", self._version, model_path)
            return {"loaded": True, "model_version": self._version, "loaded_at": self._loaded_at}

    @property
    def model(self):
        return self._model

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)

    @property
    def loaded_at(self) -> str | None:
        return self._loaded_at

    def is_loaded(self) -> bool:
        return self._model is not None


store = ModelStore()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.config["JSON_SORT_KEYS"] = False
    register_routes(flask_app)
    register_error_handlers(flask_app)

    # Best-effort warm load. A failure here must not stop the process from
    # booting -- the pod still needs to pass its probes and serve /health.
    try:
        store.load()
    except Exception as exc:  # noqa: BLE001
        log.error("Initial model load failed: %s", exc)

    return flask_app


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised for a malformed client payload. Surfaces as HTTP 400."""


def _parse_json_body() -> dict:
    """Return the request's JSON object, treating absent/empty bodies as {}.

    POST /predict is documented as taking *optional* parameters, so an empty
    body is valid. A body that is present but not a JSON object is not.
    """
    if not request.data or not request.data.strip():
        return {}

    # silent=True so a syntax error becomes our 400, not Werkzeug's HTML 400.
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError("request body is not valid JSON; send a JSON object or an empty body")
    if not isinstance(payload, dict):
        raise ValidationError(f"request body must be a JSON object, got {type(payload).__name__}")
    return payload


def _validate_forecast_days(payload: dict) -> int:
    """Pull and bounds-check forecast_days, defaulting to 7."""
    raw = payload.get("forecast_days", config.FORECAST_DAYS)

    if raw is None:
        raise ValidationError("forecast_days cannot be null")
    # Reject bools explicitly: isinstance(True, int) is True in Python.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValidationError(f"forecast_days must be a number, got {type(raw).__name__}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValidationError(f"forecast_days must be a whole number, got {raw}")

    days = int(raw)
    if days < 1 or days > MAX_FORECAST_DAYS:
        raise ValidationError(
            f"forecast_days must be between 1 and {MAX_FORECAST_DAYS}, got {days}"
        )
    return days


def _validate_history(raw_history) -> pd.DataFrame:
    """Validate a caller-supplied price history into a Prophet ds/y frame.

    Accepts a list of objects keyed {date|ds} and {close|y|price}.
    """
    if not isinstance(raw_history, list):
        raise ValidationError(
            f"historical_data must be a list of objects, got {type(raw_history).__name__}"
        )
    if len(raw_history) < MIN_HISTORY_POINTS:
        raise ValidationError(
            f"historical_data needs at least {MIN_HISTORY_POINTS} points to fit a model, "
            f"got {len(raw_history)}"
        )
    if len(raw_history) > MAX_HISTORY_POINTS:
        raise ValidationError(
            f"historical_data is capped at {MAX_HISTORY_POINTS} points, got {len(raw_history)}"
        )

    rows = []
    for i, item in enumerate(raw_history):
        if not isinstance(item, dict):
            raise ValidationError(
                f"historical_data[{i}] must be an object, got {type(item).__name__}"
            )

        date_val = item.get("date", item.get("ds"))
        close_val = item.get("close", item.get("y", item.get("price")))

        if date_val is None:
            raise ValidationError(f"historical_data[{i}] is missing a 'date' (or 'ds') field")
        if close_val is None:
            raise ValidationError(f"historical_data[{i}] is missing a 'close' (or 'y') field")
        if isinstance(close_val, bool) or not isinstance(close_val, (int, float)):
            raise ValidationError(
                f"historical_data[{i}].close must be a number, got {type(close_val).__name__}"
            )
        if close_val <= 0:
            raise ValidationError(f"historical_data[{i}].close must be positive, got {close_val}")

        try:
            parsed_date = pd.to_datetime(date_val)
        except Exception:  # noqa: BLE001
            raise ValidationError(
                f"historical_data[{i}].date is not a parseable date: {date_val!r}"
            )

        rows.append({"ds": parsed_date, "y": float(close_val)})

    df = pd.DataFrame(rows).drop_duplicates(subset="ds", keep="last")
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < MIN_HISTORY_POINTS:
        raise ValidationError(
            f"historical_data has only {len(df)} unique dates after deduplication; "
            f"at least {MIN_HISTORY_POINTS} are required"
        )
    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_routes(flask_app: Flask) -> None:
    @flask_app.get("/health")
    def health():
        """Liveness + readiness probe. 200 whenever the process is serving."""
        try:
            return (
                jsonify(
                    {
                        "status": "ok",
                        "uptime_seconds": round(time.time() - START_TIME, 2),
                        "model_version": store.version,
                        "model_loaded": store.is_loaded(),
                        "model_loaded_at": store.loaded_at,
                        "model_stage": os.getenv("MODEL_STAGE", "development"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ),
                200,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("health check failed")
            return jsonify({"status": "error", "error": str(exc)}), 500

    @flask_app.post("/predict")
    def predict():
        """Forecast the next N daily closes (default 7).

        Body (all optional):
            {"forecast_days": 7}
            {"historical_data": [{"date": "2026-01-01", "close": 42000.0}, ...]}

        With historical_data, a fresh Prophet model is fitted to the supplied
        series instead of using the registered artifact.
        """
        try:
            payload = _parse_json_body()
            days = _validate_forecast_days(payload)
            raw_history = payload.get("historical_data")

            if raw_history is not None:
                history = _validate_history(raw_history)
                log.info("Fitting ad-hoc model on %d caller-supplied points", len(history))
                model, forecast = train_forecast.fit_and_forecast(history, days)
                source = "custom_historical_data"
                model_version = None
            else:
                if not store.is_loaded():
                    return (
                        jsonify(
                            {
                                "error": "Service Unavailable",
                                "message": (
                                    "no model artifact is loaded; run the training pipeline "
                                    "then POST /reload"
                                ),
                            }
                        ),
                        503,
                    )
                model = store.model
                future = model.make_future_dataframe(periods=days, freq="D", include_history=False)
                forecast = model.predict(future)
                source = "registered_model"
                model_version = store.version

            predictions = train_forecast.forecast_to_records(forecast)

            return (
                jsonify(
                    {
                        "forecast_days": days,
                        "source": source,
                        "model_version": model_version,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "predictions": predictions,
                    }
                ),
                200,
            )

        except ValidationError as exc:
            log.warning("400 on /predict: %s", exc)
            return jsonify({"error": "Bad Request", "message": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            log.exception("500 on /predict")
            return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

    @flask_app.get("/metrics")
    def metrics():
        """Validation metrics (MAE / RMSE / MAPE) from the latest training run."""
        try:
            data = store.metrics
            if not data:
                return (
                    jsonify(
                        {
                            "error": "Not Found",
                            "message": "no validation metrics available; train a model first",
                        }
                    ),
                    404,
                )

            return (
                jsonify(
                    {
                        "model_version": data.get("model_version"),
                        "model_type": data.get("model_type", "prophet"),
                        "trained_at": data.get("trained_at"),
                        "evaluation_window": {
                            "start": data.get("test_start"),
                            "end": data.get("test_end"),
                            "days": data.get("horizon_days"),
                        },
                        "metrics": {
                            "mae": data.get("mae"),
                            "rmse": data.get("rmse"),
                            "mape": data.get("mape"),
                        },
                        "threshold": {
                            "mape_threshold": data.get("mape_threshold"),
                            "passed": data.get("threshold_passed"),
                        },
                        "train_rows": data.get("train_rows"),
                        "train_window_days": data.get("train_window_days"),
                    }
                ),
                200,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("500 on /metrics")
            return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

    @flask_app.post("/reload")
    def reload_model():
        """Re-read the model artifact from disk.

        Called by the Airflow DAG's final task once a retrained model has been
        registered, so running pods pick it up without a restart.
        """
        try:
            previous = store.version
            result = store.load()

            if not result.get("loaded"):
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": result.get("reason", "model could not be loaded"),
                            "previous_version": previous,
                        }
                    ),
                    503,
                )

            return (
                jsonify(
                    {
                        "status": "reloaded",
                        "previous_version": previous,
                        "model_version": result["model_version"],
                        "changed": previous != result["model_version"],
                        "loaded_at": result["loaded_at"],
                    }
                ),
                200,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("500 on /reload")
            return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500

    @flask_app.get("/")
    def index():
        """Tiny service descriptor, handy when poking the Ingress by hand."""
        return (
            jsonify(
                {
                    "service": "btc-forecast-api",
                    "model": "prophet",
                    "endpoints": {
                        "GET /health": "status, uptime, loaded model version",
                        "POST /predict": "7-day BTC close forecast",
                        "GET /metrics": "MAE / RMSE / MAPE from latest validation",
                        "POST /reload": "re-read model artifact from disk",
                    },
                }
            ),
            200,
        )


def register_error_handlers(flask_app: Flask) -> None:
    """JSON (never HTML) for the errors Flask raises before a route runs."""

    @flask_app.errorhandler(400)
    def bad_request(exc):
        return (
            jsonify({"error": "Bad Request", "message": getattr(exc, "description", str(exc))}),
            400,
        )

    @flask_app.errorhandler(404)
    def not_found(exc):
        return jsonify({"error": "Not Found", "message": f"no such endpoint: {request.path}"}), 404

    @flask_app.errorhandler(405)
    def method_not_allowed(exc):
        return (
            jsonify(
                {
                    "error": "Method Not Allowed",
                    "message": f"{request.method} is not allowed on {request.path}",
                }
            ),
            405,
        )

    @flask_app.errorhandler(500)
    def server_error(exc):
        log.exception("unhandled 500")
        return jsonify({"error": "Internal Server Error", "message": str(exc)}), 500


# Module-level WSGI callable: `gunicorn --bind 0.0.0.0:5000 app:app`
app = create_app()


if __name__ == "__main__":
    # Development server only. Production uses gunicorn (see app/Dockerfile).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
