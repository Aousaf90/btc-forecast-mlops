#!/usr/bin/env bash
# Capture Part A evidence (Flask REST API, 10 marks).
#
# Writes a transcript of every step to screenshots/logs/ as insurance, and
# prints each one to the terminal so it can be screenshotted. Run from the
# repo root:   bash scripts/capture_part_a.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
PY="$ROOT/venv/bin/python"
LOGS="$ROOT/screenshots/logs"
mkdir -p "$LOGS"

# Replace the absolute home path with ~ in every transcript, so the published
# logs carry no local filesystem identity.
scrub () { sed -i "s|$HOME|~|g" "$LOGS"/*.txt 2>/dev/null || true; }
banner () { printf '\n\033[1;36m%s\033[0m\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 ${#1}))"; }
jq_ () { "$PY" -m json.tool; }

# ---------------------------------------------------------------- A4: training
banner "A4  MODEL TRAINING - MAE / RMSE / MAPE on a held-out 7-day window"
"$PY" -m src.train_forecast 2>&1 | grep -viE "plotly|cmdstanpy" | tee "$LOGS/A4_training.txt" | tail -32

# ---------------------------------------------------------------- A5: backtest
banner "A5  ROLLING-ORIGIN BACKTEST - tuned vs untuned baseline"
"$PY" -m src.backtest --compare 2>&1 | grep -viE "plotly|INFO" | tee "$LOGS/A5_backtest.txt" | tail -45

# ---------------------------------------------------------------- A3: pytest
banner "A3  PYTEST SUITE - unit tests for every endpoint"
"$PY" -m pytest tests/ -v -p no:warnings 2>&1 | tee "$LOGS/A3_pytest.txt" | tail -45

# ---------------------------------------------------------- start gunicorn
banner "A3  GUNICORN - production WSGI server"
MODEL_DIR="$ROOT/models" DATA_DIR="$ROOT/data" "$ROOT/venv/bin/gunicorn" \
    --chdir app --bind 0.0.0.0:5000 --workers 2 --timeout 120 \
    app:app > "$LOGS/A3_gunicorn.txt" 2>&1 &
GPID=$!
trap 'kill -TERM $GPID 2>/dev/null' EXIT

for _ in $(seq 1 60); do
  curl -sf localhost:5000/health 2>/dev/null | grep -q '"model_loaded": true' && break
  sleep 0.5
done
cat "$LOGS/A3_gunicorn.txt"
echo "  (gunicorn --bind 0.0.0.0:5000 app:app  |  2 workers, PID $GPID)"

# ---------------------------------------------------------------- A1: endpoints
banner "A1  GET /health"
curl -s localhost:5000/health | tee "$LOGS/A1_health.txt" | jq_

banner "A1  GET /metrics"
curl -s localhost:5000/metrics | tee "$LOGS/A1_metrics.txt" | jq_

banner "A1  POST /predict   (default: 7 days)"
curl -s -X POST localhost:5000/predict -H 'Content-Type: application/json' -d '{}' \
  | tee "$LOGS/A1_predict.txt" | jq_

banner "A1  POST /reload   (used by the Airflow DAG)"
curl -s -X POST localhost:5000/reload | tee "$LOGS/A1_reload.txt" | jq_

# ---------------------------------------------------- A2: validation / errors
banner "A2  INPUT VALIDATION & ERROR HANDLING"
{
  for payload in '{"forecast_days":-1}' '{"forecast_days":"seven"}' \
                 '{"forecast_days":null}' '{bad json' '[1,2,3]'; do
    printf '\n$ curl -X POST /predict -d %s\n  -> ' "$payload"
    curl -s -o /tmp/_r -w 'HTTP %{http_code}  ' -X POST localhost:5000/predict \
         -H 'Content-Type: application/json' -d "$payload"
    cat /tmp/_r; echo
  done
  printf '\n$ curl /no-such-endpoint\n  -> '
  curl -s -o /tmp/_r -w 'HTTP %{http_code}  ' localhost:5000/no-such-endpoint; cat /tmp/_r; echo
  printf '\n$ curl -X POST /health   (wrong method)\n  -> '
  curl -s -o /tmp/_r -w 'HTTP %{http_code}  ' -X POST localhost:5000/health; cat /tmp/_r; echo
} | tee "$LOGS/A2_validation.txt"

kill -TERM $GPID 2>/dev/null
scrub
banner "DONE - transcripts saved to screenshots/logs/"
ls -1 "$LOGS" | sed 's/^/  /'
