# Verification Runbook

How to test the whole solution yourself, then record the demo. Maps each Bloc 4
grading criterion to a concrete check.

## 1. Unit tests (fast, no Docker)

The repo ships a Python 3.12 test virtualenv with PyTorch already installed
(`.venv-test/`, git-ignored). Run the full suite:

```bash
cd MLops
.venv-test/Scripts/python.exe -m pytest -q          # Windows (Git Bash)
# or, fresh on any machine:
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt
PYTHONPATH=. .venv/Scripts/python.exe -m pytest -q
```

Expected: **16 passed**. Covers the model factory (ResNet18 + CNN), the API
endpoints, the drift engine, the dataloaders, and the `evaluate()` metrics.

Drift report sample (no stack needed):

```bash
PYTHONPATH=. .venv-test/Scripts/python.exe -m monitoring.drift.generate_sample_report
# -> monitoring/drift/reports/drift_report.html (+ .json + .prom)
```

## 2. Full stack (Docker) - end to end

```bash
cd MLops
docker compose build           # rebuild to pick up the latest code
docker compose up airflow-init # once
docker compose up -d
docker compose ps              # wait until healthy (~60s)
```

URLs: Airflow http://localhost:8080 (admin/admin) - FastAPI
http://localhost:8000/docs - Streamlit http://localhost:8501 - MLflow
http://localhost:5500 - MinIO http://localhost:9001 (minioadmin/minioadmin) -
Prometheus http://localhost:9090 - Grafana http://localhost:3000 (admin/admin).

Trigger the data pipeline in Airflow (`dandelion_data_pipeline`), wait ~5-7 min.

## 3. Criterion-by-criterion checklist

| Criterion (weight) | How to verify |
|---|---|
| Specs / business relevance (15%) | `deliverables/cahier_des_charges.docx` (KPIs, scope, risks) |
| ML model quality (25%) | MLflow run shows `arch=resnet18`, `val_accuracy`, `val_f1`; notebooks under `notebooks/` |
| Serving API (15%) | `GET /version` shows `model_arch`; `POST /predict` returns class + probabilities; `POST /predict` on a text file returns 400 |
| CI/CD (20%) | GitHub Actions tab: green run with `tests` -> `build` -> `deploy` |
| Automated retraining (10%) | Airflow `dandelion_retrain_pipeline` graph: `sync -> check_drift -> train -> refresh_baseline`; or `python -m retrain.retrain --data-dir data/processed --reference-dir data/baseline` |
| Production monitoring (10%) | Grafana "Dandelion Classifier" (latency/throughput) + "Dandelion Data Drift" (PSI) dashboards; `monitoring/drift/reports/drift_report.html` |
| Presentation (5%) | `deliverables/presentation.pptx` (18 slides) |

Quick API check once a model exists:

```bash
curl -o test_dandelion.jpg "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg"
curl -s http://localhost:8000/version
curl -s -F "file=@test_dandelion.jpg" http://localhost:8000/predict
```

## 4. Record the demo (Loom, 5 min)

Follow `docs/LOOM_SCRIPT.md` - it has the timed storyboard and the exact flow:
Prediction API -> CI/CD -> Monitoring (drift, metrics) -> Retraining.

Pre-flight: stack up, one training run done so a model is served, both
dashboards open, `drift_report.html` open, two test images downloaded.

## 5. Notes

- The Docker image uses the pinned versions in `requirements.txt` (Python 3.10 +
  torch 1.13). The `.venv-test/` is only for fast local unit testing.
- The API exposes drift gauges from `monitoring/drift/reports/drift_metrics.json`
  (a committed sample ships so Grafana shows data before the first real run).
- Optional richer drift reports: `pip install -r requirements-monitoring.txt`
  (Evidently); the engine falls back to a self-rendered HTML otherwise.
- Stop: `docker compose stop` (keep data) or `docker compose down -v` (wipe).
