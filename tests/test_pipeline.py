"""Unit tests for the ETL, feature-engineering and training modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_price_series


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_forecast_scores_zero(self):
        from src.train_forecast import compute_metrics

        y = np.array([100.0, 200.0, 300.0])
        assert compute_metrics(y, y) == {"mae": 0.0, "rmse": 0.0, "mape": 0.0}

    def test_known_values(self):
        from src.train_forecast import compute_metrics

        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 180.0])
        m = compute_metrics(actual, pred)
        assert m["mae"] == pytest.approx(15.0)
        assert m["rmse"] == pytest.approx(np.sqrt((100 + 400) / 2))
        # |10/100| = 10%, |-20/200| = 10%  -> mean 10%
        assert m["mape"] == pytest.approx(10.0)

    def test_rmse_penalises_outliers_more_than_mae(self):
        from src.train_forecast import compute_metrics

        actual = np.array([100.0, 100.0, 100.0, 100.0])
        m = compute_metrics(actual, np.array([100.0, 100.0, 100.0, 180.0]))
        assert m["rmse"] > m["mae"]

    def test_rejects_shape_mismatch(self):
        from src.train_forecast import compute_metrics

        with pytest.raises(ValueError, match="shape mismatch"):
            compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_rejects_empty_window(self):
        from src.train_forecast import compute_metrics

        with pytest.raises(ValueError, match="empty"):
            compute_metrics(np.array([]), np.array([]))

    def test_zero_actual_does_not_produce_inf_mape(self):
        """A zero actual must not silently poison the registration gate."""
        from src.train_forecast import compute_metrics

        m = compute_metrics(np.array([0.0, 100.0]), np.array([5.0, 110.0]))
        assert np.isfinite(m["mape"])
        assert m["mape"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_produces_prophet_column_names(self, isolated_env, raw_df):
        from src import features

        out = features.build_features(raw_df)
        assert "ds" in out.columns and "y" in out.columns
        assert pd.api.types.is_datetime64_any_dtype(out["ds"])
        assert pd.api.types.is_float_dtype(out["y"])

    def test_computes_required_indicators(self, isolated_env, raw_df):
        """The rubric names 7-/30-day moving averages and volatility."""
        from src import features

        out = features.build_features(raw_df)
        for col in ("ma_7", "ma_30", "volatility_7", "volatility_30"):
            assert col in out.columns, f"missing required indicator {col}"
            assert out[col].notna().all(), f"{col} has nulls after warm-up drop"

    def test_moving_average_is_arithmetically_correct(self, isolated_env, raw_df):
        from src import features

        out = features.build_features(raw_df, drop_warmup=False)
        idx = 50
        expected = out["y"].iloc[idx - 6 : idx + 1].mean()
        assert out["ma_7"].iloc[idx] == pytest.approx(expected)

    def test_volatility_is_annualised_std_of_log_returns(self, isolated_env, raw_df):
        from src import features

        out = features.build_features(raw_df, drop_warmup=False)
        idx = 60
        expected = out["log_return"].iloc[idx - 6 : idx + 1].std() * np.sqrt(365)
        assert out["volatility_7"].iloc[idx] == pytest.approx(expected)

    def test_drops_warmup_rows(self, isolated_env, raw_df):
        from src import features

        kept = features.build_features(raw_df, drop_warmup=True)
        full = features.build_features(raw_df, drop_warmup=False)
        assert len(full) - len(kept) == 30

    def test_fills_calendar_gaps(self, isolated_env, raw_df):
        """BTC trades daily, so a missing day is a data gap, not a weekend."""
        from src import features

        gappy = raw_df.drop(index=[40, 41, 42]).reset_index(drop=True)
        out = features.build_features(gappy, drop_warmup=False)
        assert (out["ds"].diff().dropna() == pd.Timedelta(days=1)).all()

    def test_drops_nonpositive_prices(self, isolated_env, raw_df):
        from src import features

        dirty = raw_df.copy()
        dirty.loc[25, "Close"] = 0.0
        dirty.loc[26, "Close"] = np.nan
        out = features.build_features(dirty, drop_warmup=False)
        assert (out["y"] > 0).all()

    def test_deduplicates_dates(self, isolated_env, raw_df):
        from src import features

        dupes = pd.concat([raw_df, raw_df.iloc[[10, 11]]], ignore_index=True)
        out = features.build_features(dupes, drop_warmup=False)
        assert out["ds"].is_unique


# ---------------------------------------------------------------------------
# ETL / providers
# ---------------------------------------------------------------------------


class TestEtl:
    def test_drops_todays_partial_candle(self, isolated_env):
        """An in-progress day has an unsettled close and must not be trained on."""
        from src.etl_fetch import drop_partial_candle

        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        df = pd.DataFrame({"Date": [today - pd.Timedelta(days=1), today], "Close": [1.0, 2.0]})
        out = drop_partial_candle(df)
        assert len(out) == 1
        assert out["Date"].max() < today

    def test_keeps_all_rows_when_none_are_partial(self, isolated_env):
        from src.etl_fetch import drop_partial_candle

        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        df = pd.DataFrame({"Date": [today - pd.Timedelta(days=d) for d in (3, 2, 1)]})
        assert len(drop_partial_candle(df)) == 3

    def test_cache_merge_prefers_fresh_rows(self, isolated_env, tmp_path):
        from src.etl_fetch import merge_with_cache

        cache = tmp_path / "raw.csv"
        pd.DataFrame(
            {"Date": pd.date_range("2026-01-01", periods=3), "Close": [1.0, 2.0, 3.0]}
        ).to_csv(cache, index=False)

        fresh = pd.DataFrame(
            {"Date": pd.date_range("2026-01-03", periods=3), "Close": [99.0, 4.0, 5.0]}
        )
        out = merge_with_cache(fresh, cache)

        assert len(out) == 5, "overlapping dates should merge, not duplicate"
        assert out.loc[out["Date"] == "2026-01-03", "Close"].iloc[0] == 99.0

    def test_cache_merge_without_existing_cache(self, isolated_env, tmp_path):
        from src.etl_fetch import merge_with_cache

        fresh = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=2), "Close": [1.0, 2.0]})
        pd.testing.assert_frame_equal(merge_with_cache(fresh, tmp_path / "absent.csv"), fresh)

    def test_load_cache_raises_when_absent(self, isolated_env, tmp_path):
        from src.etl_fetch import DataFetchError, load_cache

        with pytest.raises(DataFetchError, match="no cache exists"):
            load_cache(tmp_path / "nope.csv")

    def test_fails_over_to_the_next_provider(self, isolated_env, monkeypatch):
        """yfinance being rate-limited must not end the run."""
        from src import etl_fetch, providers

        calls = []

        def dead(ticker, period):
            calls.append("yfinance")
            raise providers.ProviderError("HTTP 429")

        def alive(ticker, period):
            calls.append("binance")
            return pd.DataFrame(
                {
                    "Date": pd.date_range("2026-01-01", periods=5),
                    "Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0,
                }
            )

        monkeypatch.setitem(providers.PROVIDERS, "yfinance", dead)
        monkeypatch.setitem(providers.PROVIDERS, "binance", alive)
        monkeypatch.setattr(etl_fetch.time, "sleep", lambda *_: None)

        df, used = etl_fetch.fetch_btc_data(
            "BTC-USD", "1y", provider_chain=("yfinance", "binance"), max_attempts=2
        )
        assert used == "binance"
        assert calls == ["yfinance", "yfinance", "binance"], "should retry then fail over"
        assert len(df) == 5

    def test_raises_when_every_provider_fails(self, isolated_env, monkeypatch):
        from src import etl_fetch, providers

        def dead(ticker, period):
            raise providers.ProviderError("down")

        monkeypatch.setitem(providers.PROVIDERS, "yfinance", dead)
        monkeypatch.setattr(etl_fetch.time, "sleep", lambda *_: None)

        with pytest.raises(etl_fetch.DataFetchError, match="all providers failed"):
            etl_fetch.fetch_btc_data("BTC-USD", "1y", provider_chain=("yfinance",), max_attempts=1)

    @pytest.mark.parametrize(
        "period,days", [("1d", 1), ("2wk", 14), ("6mo", 180), ("5y", 1825), ("max", 5475)]
    )
    def test_period_parsing(self, period, days):
        from src.providers import period_to_days

        assert period_to_days(period) == days

    def test_rejects_unknown_period(self):
        from src.providers import period_to_days

        with pytest.raises(ValueError, match="unrecognised period"):
            period_to_days("banana")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTraining:
    def test_train_window_keeps_most_recent_rows(self, isolated_env):
        from src.train_forecast import apply_train_window

        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=500), "y": range(500)})
        out = apply_train_window(df, 365)
        assert len(out) == 365
        assert out["y"].iloc[-1] == 499, "must keep the newest rows, not the oldest"
        assert out["y"].iloc[0] == 135

    @pytest.mark.parametrize("window", [0, None, 10_000])
    def test_train_window_passthrough(self, isolated_env, window):
        from src.train_forecast import apply_train_window

        df = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=100), "y": range(100)})
        assert len(apply_train_window(df, window)) == 100

    def test_registers_and_writes_all_artifacts(self, trained_model_dir):
        cfg = trained_model_dir
        assert cfg.MODEL_JSON.exists(), "prophet_model.json not written"
        assert cfg.METRICS_JSON.exists(), "metrics.json not written"
        assert cfg.FORECAST_JSON.exists(), "forecast.json not written"
        assert list(cfg.RUNS_DIR.glob("*/run.json")), "no run tracked in RUNS_DIR"

    def test_metrics_payload_is_complete(self, trained_model_dir):
        import json

        data = json.loads(trained_model_dir.METRICS_JSON.read_text())
        for key in ("mae", "rmse", "mape", "model_version", "trained_at", "registered"):
            assert key in data, f"metrics.json missing {key}"
        assert data["registered"] is True
        assert len(data["forecast"]) == 7

    def test_model_survives_json_round_trip(self, trained_model_dir):
        """The exact Airflow-writes / Flask-reads path. Never pickle a Prophet model."""
        from src import train_forecast

        model = train_forecast.load_model(trained_model_dir.MODEL_JSON)
        future = model.make_future_dataframe(periods=7, freq="D", include_history=False)
        reloaded = train_forecast.forecast_to_records(model.predict(future))

        import json

        saved = json.loads(trained_model_dir.FORECAST_JSON.read_text())["forecast"]
        for a, b in zip(saved, reloaded):
            assert a["date"] == b["date"]
            assert a["predicted_close"] == pytest.approx(b["predicted_close"], abs=0.01)

    def test_threshold_gate_rejects_a_bad_model(self, isolated_env, raw_df):
        """An unreachable threshold must leave the live artifact untouched."""
        from src import features, train_forecast

        feats = features.build_features(raw_df)
        feats.to_csv(isolated_env.FEATURES_CSV, index=False)

        result = train_forecast.train(
            features_path=isolated_env.FEATURES_CSV, horizon=7, mape_threshold=0.0001
        )
        assert result["registered"] is False
        assert result["threshold_passed"] is False
        assert not isolated_env.MODEL_JSON.exists(), "rejected model must not be registered"

    def test_force_register_overrides_the_gate(self, isolated_env, raw_df):
        from src import features, train_forecast

        feats = features.build_features(raw_df)
        feats.to_csv(isolated_env.FEATURES_CSV, index=False)

        result = train_forecast.train(
            features_path=isolated_env.FEATURES_CSV,
            horizon=7,
            mape_threshold=0.0001,
            force_register=True,
        )
        assert result["threshold_passed"] is False
        assert result["registered"] is True
        assert isolated_env.MODEL_JSON.exists()

    def test_test_window_is_never_in_the_training_data(self, isolated_env, raw_df):
        """Leakage guard: the held-out week must be excluded from the fit."""
        from src import features, train_forecast

        feats = features.build_features(raw_df)
        feats.to_csv(isolated_env.FEATURES_CSV, index=False)

        result = train_forecast.train(
            features_path=isolated_env.FEATURES_CSV, horizon=7, mape_threshold=100.0
        )
        assert pd.Timestamp(result["train_end"]) < pd.Timestamp(result["test_start"])

    def test_refuses_to_train_on_too_little_data(self, isolated_env):
        from src import train_forecast

        tiny = pd.DataFrame({"ds": pd.date_range("2026-01-01", periods=20), "y": range(20)})
        tiny.to_csv(isolated_env.FEATURES_CSV, index=False)

        with pytest.raises(ValueError, match="need at least"):
            train_forecast.train(features_path=isolated_env.FEATURES_CSV, horizon=7)

    def test_forecast_records_are_ordered_and_bounded(self, trained_model_dir):
        import json

        rows = json.loads(trained_model_dir.FORECAST_JSON.read_text())["forecast"]
        dates = [pd.Timestamp(r["date"]) for r in rows]
        assert dates == sorted(dates)
        for r in rows:
            assert r["lower_bound"] <= r["predicted_close"] <= r["upper_bound"]
