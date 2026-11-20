"""Endpoint tests for the Flask API (Part A3: pytest coverage of every route)."""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200_and_status_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"

    def test_reports_uptime_and_model_version(self, client):
        body = client.get("/health").get_json()
        assert isinstance(body["uptime_seconds"], (int, float))
        assert body["uptime_seconds"] >= 0
        assert body["model_loaded"] is True
        assert body["model_version"]

    def test_stays_200_without_a_model(self, client_no_model):
        """Critical: both k8s probes point here, so a missing artifact must not
        make the liveness probe restart-loop a healthy pod."""
        resp = client_no_model.get("/health")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["model_version"] is None

    def test_rejects_post(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405
        assert resp.get_json()["error"] == "Method Not Allowed"


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_defaults_to_seven_days(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["forecast_days"] == 7
        assert len(body["predictions"]) == 7

    def test_empty_body_is_valid(self, client):
        """Parameters are documented as optional, so no body is a valid request."""
        resp = client.post("/predict")
        assert resp.status_code == 200
        assert len(resp.get_json()["predictions"]) == 7

    def test_prediction_shape_and_types(self, client):
        body = client.post("/predict", json={}).get_json()
        assert body["source"] == "registered_model"
        assert body["model_version"]

        for row in body["predictions"]:
            assert set(row) == {"date", "predicted_close", "lower_bound", "upper_bound"}
            assert isinstance(row["predicted_close"], (int, float))
            assert row["predicted_close"] > 0
            assert row["lower_bound"] <= row["predicted_close"] <= row["upper_bound"]

    def test_dates_are_consecutive_and_future(self, client):
        import pandas as pd

        rows = client.post("/predict", json={}).get_json()["predictions"]
        dates = [pd.Timestamp(r["date"]) for r in rows]
        deltas = {(b - a).days for a, b in zip(dates, dates[1:])}
        assert deltas == {1}, f"forecast dates are not consecutive: {dates}"

    @pytest.mark.parametrize("days", [1, 3, 14, 30])
    def test_honours_forecast_days(self, client, days):
        resp = client.post("/predict", json={"forecast_days": days})
        assert resp.status_code == 200
        assert len(resp.get_json()["predictions"]) == days

    def test_accepts_custom_history(self, client, sample_history):
        resp = client.post("/predict", json={"historical_data": sample_history})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["source"] == "custom_historical_data"
        assert len(body["predictions"]) == 7

    def test_503_when_no_model_loaded(self, client_no_model):
        resp = client_no_model.post("/predict", json={})
        assert resp.status_code == 503
        assert "no model" in resp.get_json()["message"].lower()

    # --- 400-level input validation (Part A2) ------------------------------

    @pytest.mark.parametrize(
        "payload,fragment",
        [
            ({"forecast_days": 0}, "between 1"),
            ({"forecast_days": -5}, "between 1"),
            ({"forecast_days": 500}, "between 1"),
            ({"forecast_days": None}, "cannot be null"),
            ({"forecast_days": "seven"}, "must be a number"),
            ({"forecast_days": True}, "must be a number"),
            ({"forecast_days": 7.5}, "whole number"),
        ],
    )
    def test_rejects_bad_forecast_days(self, client, payload, fragment):
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"] == "Bad Request"
        assert fragment in body["message"]

    def test_rejects_malformed_json(self, client):
        resp = client.post("/predict", data="{not valid json", content_type="application/json")
        assert resp.status_code == 400
        assert "valid JSON" in resp.get_json()["message"]

    def test_rejects_non_object_body(self, client):
        resp = client.post("/predict", data=json.dumps([1, 2, 3]), content_type="application/json")
        assert resp.status_code == 400
        assert "must be a JSON object" in resp.get_json()["message"]

    def test_rejects_null_body(self, client):
        resp = client.post("/predict", data="null", content_type="application/json")
        assert resp.status_code == 400

    @pytest.mark.parametrize(
        "history,fragment",
        [
            ("not-a-list", "must be a list"),
            ([], "at least 30"),
            ([{"date": "2026-01-01", "close": 100.0}] * 5, "at least 30"),
        ],
    )
    def test_rejects_bad_history_shape(self, client, history, fragment):
        resp = client.post("/predict", json={"historical_data": history})
        assert resp.status_code == 400
        assert fragment in resp.get_json()["message"]

    def test_rejects_history_missing_close(self, client, sample_history):
        broken = [dict(r) for r in sample_history]
        del broken[10]["close"]
        resp = client.post("/predict", json={"historical_data": broken})
        assert resp.status_code == 400
        assert "missing a 'close'" in resp.get_json()["message"]

    def test_rejects_history_missing_date(self, client, sample_history):
        broken = [dict(r) for r in sample_history]
        del broken[3]["date"]
        resp = client.post("/predict", json={"historical_data": broken})
        assert resp.status_code == 400
        assert "missing a 'date'" in resp.get_json()["message"]

    @pytest.mark.parametrize("bad_close", [-100.0, 0, "abc", None])
    def test_rejects_bad_close_values(self, client, sample_history, bad_close):
        broken = [dict(r) for r in sample_history]
        broken[20]["close"] = bad_close
        resp = client.post("/predict", json={"historical_data": broken})
        assert resp.status_code == 400

    def test_rejects_unparseable_date(self, client, sample_history):
        broken = [dict(r) for r in sample_history]
        broken[15]["date"] = "not-a-date"
        resp = client.post("/predict", json={"historical_data": broken})
        assert resp.status_code == 400
        assert "parseable date" in resp.get_json()["message"]


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_returns_mae_rmse_mape(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        metrics = resp.get_json()["metrics"]
        assert set(metrics) == {"mae", "rmse", "mape"}
        for name, value in metrics.items():
            assert isinstance(value, (int, float)), f"{name} is not numeric"
            assert value >= 0

    def test_includes_version_and_eval_window(self, client):
        body = client.get("/metrics").get_json()
        assert body["model_version"]
        assert body["model_type"] == "prophet"
        assert body["evaluation_window"]["days"] == 7
        assert body["evaluation_window"]["start"]
        assert body["evaluation_window"]["end"]

    def test_reports_threshold_outcome(self, client):
        threshold = client.get("/metrics").get_json()["threshold"]
        assert threshold["passed"] is True
        assert isinstance(threshold["mape_threshold"], (int, float))

    def test_404_when_no_metrics(self, client_no_model):
        resp = client_no_model.get("/metrics")
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "Not Found"

    def test_rejects_post(self, client):
        assert client.post("/metrics").status_code == 405


# ---------------------------------------------------------------------------
# POST /reload  (required by the Airflow DAG's reload-trigger task, Part C1)
# ---------------------------------------------------------------------------


class TestReload:
    def test_reloads_successfully(self, client):
        resp = client.post("/reload")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "reloaded"
        assert body["model_version"]

    def test_picks_up_a_newly_written_model(self, client, trained_model_dir):
        """The actual Airflow -> Flask contract: retrain on disk, then /reload."""
        before = client.post("/reload").get_json()["model_version"]

        from src import train_forecast

        train_forecast.train(
            features_path=trained_model_dir.FEATURES_CSV,
            horizon=7,
            mape_threshold=100.0,
        )

        after = client.post("/reload").get_json()
        assert after["status"] == "reloaded"
        assert after["model_version"] != before
        assert after["previous_version"] == before
        assert after["changed"] is True

    def test_503_when_artifact_missing(self, client_no_model):
        resp = client_no_model.post("/reload")
        assert resp.status_code == 503
        assert resp.get_json()["status"] == "error"

    def test_rejects_get(self, client):
        assert client.get("/reload").status_code == 405


# ---------------------------------------------------------------------------
# Routing / error contract
# ---------------------------------------------------------------------------


class TestErrorContract:
    def test_unknown_route_returns_json_404(self, client):
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404
        assert resp.is_json, "error responses must be JSON, never Flask's HTML page"
        assert resp.get_json()["error"] == "Not Found"

    def test_index_lists_endpoints(self, client):
        body = client.get("/").get_json()
        assert body["service"] == "btc-forecast-api"
        assert "POST /predict" in body["endpoints"]

    def test_all_errors_are_json(self, client):
        for resp in (
            client.get("/nope"),
            client.post("/health"),
            client.post("/predict", json={"forecast_days": -1}),
        ):
            assert resp.is_json
            assert "error" in resp.get_json()
