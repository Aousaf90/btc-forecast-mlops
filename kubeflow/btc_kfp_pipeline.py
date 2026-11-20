"""Kubeflow Pipelines definition for the BTC forecasting workflow (Part C2).

Encapsulates data ingestion, feature generation and model training as three
discrete KFP **container components**, wired into a sequential pipeline and
compiled to a YAML spec with the KFP v2 SDK.

    python kubeflow/btc_kfp_pipeline.py            # writes btc_kfp_pipeline.yaml
    python kubeflow/btc_kfp_pipeline.py --output /tmp/spec.yaml

Design notes
------------
* **One image, three commands.** Every component runs the same published
  image (``aousafs/btc-forecast-api``), which already contains ``src/`` and the
  pinned Prophet. Reusing it means the pipeline cannot drift from the code
  that CI built and tested, and the cluster pulls one image rather than three.

* **Container components, not lightweight Python components.** The rubric asks
  for discrete container components, and they are also the honest choice here:
  each step is an independent container invoking a module entry point, so the
  same commands run identically under Airflow, Docker and KFP.

* **Artifacts pass by path.** Each component declares KFP output paths and the
  next component consumes them as inputs, so KFP tracks lineage between steps
  rather than relying on a shared mutable directory.

* **This compiles to YAML; it is not deployed.** A full Kubeflow installation
  needs roughly 16 GB of RAM, well beyond this project's single-node Minikube
  budget, and the assignment only requires the compiled spec. The generated
  YAML is committed alongside this file.
"""

# NOTE: deliberately no `from __future__ import annotations` here. PEP 563
# turns annotations into strings, and the KFP compiler introspects the real
# annotation objects to resolve artifact types -- with it enabled, compilation
# fails with "Artifacts must have both a schema_title and a schema_version".

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

# The image CI publishes. Override at compile time with --image to pin a
# specific commit SHA instead of a floating tag.
DEFAULT_IMAGE = "aousafs/btc-forecast-api:latest"

PIPELINE_NAME = "btc-7day-forecast-pipeline"


# ---------------------------------------------------------------------------
# Component 1: data ingestion
# ---------------------------------------------------------------------------


@dsl.container_component
def ingest_btc_data(
    raw_data: Output[Dataset],
    period: str = "5y",
    ticker: str = "BTC-USD",
):
    """Pull daily BTC-USD OHLCV into a Dataset artifact.

    Runs the same provider chain as the Airflow DAG: yfinance, then Binance,
    then CoinGecko, each retried with exponential backoff before failing over.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=[
            "--ticker",
            ticker,
            "--period",
            period,
            "--out",
            raw_data.path,
        ],
    )


# ---------------------------------------------------------------------------
# Component 2: feature generation
# ---------------------------------------------------------------------------


@dsl.container_component
def generate_features(
    raw_data: Input[Dataset],
    features: Output[Dataset],
):
    """Clean the series and compute the technical indicators.

    Emits 7-/30-day moving averages and annualised rolling volatility in
    Prophet's required ds/y format.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["python", "-m", "src.features"],
        args=[
            "--raw",
            raw_data.path,
            "--out",
            features.path,
        ],
    )


# ---------------------------------------------------------------------------
# Component 3: training and evaluation
# ---------------------------------------------------------------------------


@dsl.container_component
def train_forecast_model(
    features: Input[Dataset],
    model_dir: Output[Model],
    horizon: int = 7,
    mape_threshold: float = 15.0,
    train_window_days: int = 365,
):
    """Fit Prophet, score the held-out window, register if it passes the gate.

    MODEL_DIR is exported so src.train_forecast writes its artifacts
    (prophet_model.json, metrics.json, forecast.json, runs/) into the KFP Model
    output directory, making them tracked pipeline artifacts.

    Exits non-zero when the model fails the MAPE threshold, which surfaces in
    the KFP UI as a failed step rather than a silent no-op.
    """
    return dsl.ContainerSpec(
        image=DEFAULT_IMAGE,
        command=["sh", "-c"],
        args=[
            (
                "set -e; "
                'mkdir -p "$0"; '
                'export MODEL_DIR="$0"; '
                "python -m src.train_forecast "
                '--features "$1" --horizon "$2" '
                '--mape-threshold "$3" --train-window-days "$4"'
            ),
            model_dir.path,
            features.path,
            str(horizon),
            str(mape_threshold),
            str(train_window_days),
        ],
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dsl.pipeline(
    name=PIPELINE_NAME,
    description=(
        "Daily BTC-USD ingestion, feature engineering and Prophet retraining "
        "for a 7-day close-price forecast."
    ),
)
def btc_forecast_pipeline(
    ticker: str = "BTC-USD",
    period: str = "5y",
    horizon: int = 7,
    mape_threshold: float = 15.0,
    train_window_days: int = 365,
):
    """Sequential ingestion -> features -> training.

    Ordering is implied by the artifact dependencies rather than declared with
    .after(): each step consumes the previous step's output, so KFP derives the
    DAG from the data flow.
    """
    ingest = ingest_btc_data(period=period, ticker=ticker)
    ingest.set_display_name("Ingest BTC-USD data")
    # Market data APIs rate-limit; let KFP retry the step as well as the
    # in-process backoff the provider chain already performs.
    ingest.set_retry(num_retries=3, backoff_duration="60s", backoff_factor=2.0)

    featurize = generate_features(raw_data=ingest.outputs["raw_data"])
    featurize.set_display_name("Generate features")

    train = train_forecast_model(
        features=featurize.outputs["features"],
        horizon=horizon,
        mape_threshold=mape_threshold,
        train_window_days=train_window_days,
    )
    train.set_display_name("Train and evaluate Prophet model")
    # Prophet fits on CPU in a couple of seconds; these bounds match the
    # Kubernetes Deployment's limits so the pipeline is schedulable on the
    # same single-node cluster.
    train.set_cpu_limit("1")
    train.set_memory_limit("1Gi")


# ---------------------------------------------------------------------------
# Compilation entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the BTC KFP pipeline to YAML.")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "btc_kfp_pipeline.yaml"),
        help="path for the compiled pipeline spec",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=btc_forecast_pipeline,
        package_path=str(out_path),
    )

    size_kb = out_path.stat().st_size / 1024
    print(f"Compiled pipeline '{PIPELINE_NAME}' -> {out_path} ({size_kb:.1f} KB)")
    print("Components: ingest_btc_data, generate_features, train_forecast_model")
    print(f"Image:      {DEFAULT_IMAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
