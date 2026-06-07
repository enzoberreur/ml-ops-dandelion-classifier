#!/usr/bin/env bash
# One-shot demo for Bloc 4 - GreenGuard MLOps dandelion classifier.
# Boots the stack, runs a live prediction, shows MLflow tracking, drift metrics,
# the monitoring dashboards, and the drift-gated retraining DAG. Built to be
# screen-recorded for the Loom (Prediction API -> tracking -> monitoring -> retrain).
#
# Usage:
#   bash demo.sh            # pause between steps (best for recording)
#   AUTO=1 bash demo.sh     # no pauses
#   bash demo.sh --open     # open dashboards in the browser
#   bash demo.sh --retrain  # also trigger the retrain DAG at the end
#
# Note: this assumes the image is already built (first build pulls PyTorch and is
# slow). If services fail to start, run: docker compose build
set -uo pipefail
cd "$(dirname "$0")"

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
AUTO="${AUTO:-0}"; OPEN=0; RETRAIN=0
for a in "$@"; do case "$a" in --open) OPEN=1;; --auto) AUTO=1;; --retrain) RETRAIN=1;; esac; done

step(){ echo; echo -e "${BLUE}==================================================${NC}"; echo -e "${BLUE}>> $1${NC}"; echo -e "${BLUE}==================================================${NC}"; }
ok(){ echo -e "${GREEN}[OK] $1${NC}"; }
warn(){ echo -e "${YELLOW}[!] $1${NC}"; }
pause(){ if [ "$AUTO" = "1" ]; then sleep "${1:-3}"; else echo; echo -e "${YELLOW}-- Press Enter to continue --${NC}"; read -r; fi; }
pp(){ if command -v jq >/dev/null 2>&1; then jq .; else python -m json.tool 2>/dev/null || cat; fi; }
open_url(){ [ "$OPEN" = "1" ] || return 0; ( cmd.exe /c start "" "$1" 2>/dev/null || xdg-open "$1" 2>/dev/null || open "$1" 2>/dev/null ) & }
wait_http(){ local url="$1" name="$2" t="${3:-180}" i=0; echo -n "Waiting for $name "; while [ "$i" -lt "$t" ]; do if curl -fsS "$url" >/dev/null 2>&1; then echo; ok "$name is up"; return 0; fi; echo -n "."; sleep 2; i=$((i+2)); done; echo; warn "$name not responding after ${t}s (continuing)"; return 1; }

command -v docker >/dev/null 2>&1 || { echo -e "${RED}docker not found on PATH. Open a terminal where 'docker' works and re-run.${NC}"; exit 1; }

step "GreenGuard MLOps - one-shot demo"
echo "Prediction API -> experiment tracking -> monitoring/drift -> retraining."
pause 1

step "1/7  Start the MLOps stack"
echo "\$ docker compose up -d"
docker compose up -d
pause

step "2/7  Wait for the prediction API, then list services"
wait_http "http://localhost:8000/health" "prediction API" 240
docker compose ps
pause

step "3/7  Which model is served? (GET /version)"
curl -fsS http://localhost:8000/version | pp
ok "Model architecture, classes and source are exposed"
pause

step "4/7  Live prediction (POST /predict)"
IMG=""
for c in test_dandelion.jpg data/test_dandelion.jpg /tmp/test_dandelion.jpg; do [ -f "$c" ] && IMG="$c" && break; done
if [ -z "$IMG" ]; then
  echo "Fetching a sample dandelion image..."
  curl -fsS "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg" -o /tmp/test_dandelion.jpg 2>/dev/null && IMG=/tmp/test_dandelion.jpg
fi
if [ -n "$IMG" ]; then
  echo "Predicting on: $IMG"
  curl -fsS -X POST http://localhost:8000/predict -F "file=@$IMG" | pp
  ok "Prediction returned (class + per-class probabilities)"
else
  warn "No test image available; use the Streamlit UI (http://localhost:8501) to upload one"
fi
pause

step "5/7  Experiment tracking (MLflow) + serving/drift metrics"
echo "MLflow UI:  http://localhost:5500   (experiment: dandelion-classifier)"
echo
echo "Drift + prediction metrics on /metrics:"
curl -fsS http://localhost:8000/metrics | grep -Ei 'drift|prediction' | head -20
open_url "http://localhost:5500"
pause

step "6/7  Monitoring dashboards + drift report"
cat <<'URLS'
  Grafana (admin/admin):  http://localhost:3000   (Dandelion Classifier + Data Drift dashboards)
  Prometheus:             http://localhost:9090
  Streamlit UI:           http://localhost:8501
  MinIO console:          http://localhost:9001   (minioadmin/minioadmin)
  Airflow:                http://localhost:8080
URLS
[ -f monitoring/drift/reports/drift_report.html ] && ok "Drift report: monitoring/drift/reports/drift_report.html"
open_url "http://localhost:3000"; open_url "http://localhost:8501"
pause

step "7/7  Automated retraining (drift-gated)"
echo "The retrain DAG only trains when incoming data has drifted:"
docker compose exec -T airflow-scheduler airflow dags list 2>/dev/null | grep -i retrain || warn "airflow not ready yet; see http://localhost:8080"
if [ "$RETRAIN" = "1" ]; then
  echo "Triggering dandelion_retrain_pipeline..."
  docker compose exec -T airflow-scheduler airflow dags trigger dandelion_retrain_pipeline 2>/dev/null && ok "Retrain DAG triggered (watch Airflow)"
fi
echo "CI/CD: https://github.com/enzoberreur/ml-ops-dandelion-classifier/actions"
pause

step "Demo complete"
echo "Stop the stack:  docker compose down"
