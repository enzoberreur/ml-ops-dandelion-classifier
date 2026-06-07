# Quick Start - Presentation Guide

A fast path to run the project for the live demo. For the narrated 5-minute
walkthrough, follow [`docs/LOOM_SCRIPT.md`](docs/LOOM_SCRIPT.md).

## Launch in 3 commands

```bash
docker compose build           # 1. build the app image
docker compose up airflow-init # 2. initialise Airflow (once)
docker compose up -d           # 3. start all services
```

Wait ~30-60 s, then check:

```bash
docker compose ps              # all services healthy
docker compose logs -f api     # API ready
```

## Service URLs

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| FastAPI | http://localhost:8000/docs | - |
| Streamlit | http://localhost:8501 | - |
| MLflow | http://localhost:5500 | - |
| MinIO | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | `admin` / `admin` |

## 10-minute demo flow

1. **Architecture (2 min)** - open `README.md`, walk the diagram: Data → Training → Serving → Monitoring.
2. **Train (3 min)** - in Airflow, trigger `dandelion_data_pipeline`
   (download → preprocess → train → push model to MinIO). While it runs, show
   **MinIO** (buckets fill) and **MLflow** (live metrics).
3. **Predict (2 min)** - once trained, upload an image in **Streamlit**, or call
   `POST /predict` in the FastAPI docs (`GET /health` → `ready`, `GET /version`).
4. **Monitoring (2 min)** - Grafana *Dandelion Classifier* (latency, throughput)
   and *Dandelion Data Drift* dashboards; open `monitoring/drift/reports/drift_report.html`.
5. **Retraining (1 min)** - show the drift-aware `dandelion_retrain_pipeline` DAG.

## Test images

```bash
curl -o test_dandelion.jpg "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/dandelion/00000001.jpg"
curl -o test_grass.jpg     "https://raw.githubusercontent.com/btphan95/greenr-airflow/refs/heads/master/data/grass/00000001.jpg"
```

## Expected results

- **Data:** ~400 images in MinIO (200 dandelion + 200 grass).
- **Model:** `best_model.pt` in bucket `dandelion-models`.
- **MLflow:** one run, accuracy ~0.90 / macro-F1 ~0.89.
- **API:** status `ready`, classes `[dandelion, grass]`.
- **Total:** ~5-7 min on CPU.

## Troubleshooting

- **Airflow won't start:** `docker compose down -v` then re-run the 3 launch commands (wipes volumes).
- **API says "Model not available":** normal before the first training run - trigger the data pipeline first, then `docker compose restart api`.
- **MinIO buckets empty:** check `docker compose logs bootstrap`; re-run `docker compose up bootstrap`.
- **Stop:** `docker compose stop` (keeps data) · `docker compose down -v` (removes everything).

## If the demo fails live

1. Show the diagrams in `README.md` (always works).
2. Walk the screenshots in `docs/screenshots/`.
3. Show the code: Airflow DAGs, PyTorch model, FastAPI app, drift module.
