"""Shared pytest fixtures.

Every test runs against a throwaway MODEL_DIR / DATA_DIR in tmp_path, so the
suite never reads or writes the developer's real artifacts and runs identically
on a clean CI checkout where models/ does not exist at all.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "app")):
    if path not in sys.path:
        sys.path.insert(0, path)


def make_price_series(days: int = 420, start: str = "2024-01-01", seed: int = 7) -> pd.DataFrame:
    """A deterministic synthetic BTC-like series: drifting geometric random walk.

    Synthetic rather than a recorded fixture so the suite needs no network and
    no checked-in market data, and stays stable as real prices move.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    shocks = rng.normal(loc=0.0012, scale=0.02, size=days)
    close = 30000.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close * (1 + rng.normal(0, 0.003, days)),
            "High": close * (1 + np.abs(rng.normal(0, 0.012, days))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.012, days))),
            "Close": close,
            "Volume": rng.uniform(1e4, 5e4, days),
        }
    )


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_price_series()


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Point config at tmp_path and reload the modules that captured its paths.

    src.config resolves its paths at import time, so the env vars must be set
    and the module reloaded before anything else imports it.
    """
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("TRAIN_WINDOW_DAYS", "365")

    from src import config as config_module

    importlib.reload(config_module)
    for name in ("src.features", "src.train_forecast", "src.etl_fetch", "src.backtest"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])

    config_module.ensure_dirs()
    return config_module


@pytest.fixture
def trained_model_dir(isolated_env, raw_df):
    """A MODEL_DIR containing a genuinely trained Prophet artifact + metrics.

    Trains a real (small) Prophet model rather than mocking, so the tests
    exercise the same model_to_json / model_from_json path that the Airflow ->
    Flask handoff depends on.
    """
    from src import features, train_forecast

    feats = features.build_features(raw_df)
    feats.to_csv(isolated_env.FEATURES_CSV, index=False)

    train_forecast.train(
        features_path=isolated_env.FEATURES_CSV,
        horizon=7,
        mape_threshold=100.0,  # synthetic data: always register, we test plumbing
    )
    assert isolated_env.MODEL_JSON.exists()
    return isolated_env


@pytest.fixture
def client(trained_model_dir):
    """Flask test client with a loaded model."""
    import app as app_module

    importlib.reload(app_module)
    app_module.app.config.update(TESTING=True)
    app_module.store.load()
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def client_no_model(isolated_env):
    """Flask test client where MODEL_DIR exists but holds no artifact."""
    import app as app_module

    importlib.reload(app_module)
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_history() -> list[dict]:
    """A caller-supplied history payload for POST /predict."""
    df = make_price_series(days=120, seed=11)
    return [
        {"date": row.Date.strftime("%Y-%m-%d"), "close": round(float(row.Close), 2)}
        for row in df.itertuples()
    ]
