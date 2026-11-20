# ---------------------------------------------------------------------------
# Airflow worker image for the BTC ETL / retraining DAG.
#
# CRITICAL: this installs the SAME shared requirements.txt as app/Dockerfile,
# so the prophet version here is identical to the API's. The Airflow container
# writes the model JSON that the Flask container deserializes on POST /reload;
# a version drift between the two breaks that handoff at runtime, not at build
# time, which makes it an expensive bug to find. One shared pin prevents it.
#
# requirements.txt deliberately excludes Flask and gunicorn (they live in
# requirements-api.txt). Airflow 2.10 pins Flask 2.2.x because
# flask-appbuilder imports flask._app_ctx_stack, removed in Flask 2.3.
# Installing the API's Flask 3.0.3 here crash-loops the scheduler with
# "ImportError: cannot import name '_app_ctx_stack'". The verification step
# below fails the build if that regression is ever reintroduced.
#
# Unlike the API image, matplotlib is NOT stripped here -- the training side is
# where plotting would be used for diagnostics, and image size matters far less
# for a worker that is not replicated across pods.
# ---------------------------------------------------------------------------
FROM apache/airflow:2.10.3-python3.10

USER root
# build-essential covers any dependency without a prebuilt wheel; curl is used
# by the DAG's reload task and for container health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Installed before the DAG/source bind mounts so this layer caches across code
# changes -- Prophet is slow to install and dominates rebuild time otherwise.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The DAG imports the pipeline modules from here (PYTHONPATH in compose).
ENV PYTHONPATH=/opt/airflow/project

# Fail the build if the Airflow image cannot load the same prophet the API
# uses, rather than discovering it when the DAG's retrain task runs.
RUN python -c "\
import prophet, pandas, numpy, flask, airflow; \
from prophet.serialize import model_to_json, model_from_json; \
from flask_appbuilder import SQLA; \
print('airflow image OK: prophet', prophet.__version__, \
      '| pandas', pandas.__version__, \
      '| flask', flask.__version__, '(airflow-compatible)')"
