# Repository Structure & Mapping to the Brief

The brief asks for two logical repositories with specific folders. This project
keeps **one repository** (simpler to deploy and demo) and maps cleanly onto both
required structures, as shown below.

## ML model code (brief: `/notebooks`, `/src`, `/tests`, `/models`, `requirements.txt`)

| Required | In this repo | Contents |
|---|---|---|
| `/notebooks` | [`notebooks/`](../notebooks/) | EDA + model-selection notebooks |
| `/src` | [`models/`](../models/) + [`app/`](../app/) | training (`models/`), serving source (`app/`) |
| `/models` | [`models/`](../models/) | architecture, training, utils |
| `/tests` | [`tests/`](../tests/) | unit + API + drift tests |
| `requirements.txt` | [`requirements.txt`](../requirements.txt) | runtime deps |

## Deployment & CI/CD code (brief: `/api`, `/k8s`, `/monitoring`, `/retrain`, `Dockerfile`, `.github/workflows/`)

| Required | In this repo | Contents |
|---|---|---|
| `/api` | [`app/api/`](../app/api/) | FastAPI serving (`/predict`, `/health`, `/version`, `/metrics`) |
| `/k8s` | [`k8s/`](../k8s/) | Minikube manifests (namespace, configmap, secret, deploys, services) |
| `/monitoring` | [`monitoring/`](../monitoring/) | Prometheus, Grafana dashboards, **drift monitoring** |
| `/retrain` | [`retrain/`](../retrain/) | drift-gated + scheduled retraining |
| `Dockerfile` | [`Dockerfile`](../Dockerfile) | single app image (API, Streamlit, MLflow, trainer) |
| `.github/workflows/` | [`.github/workflows/ci_cd.yml`](../.github/workflows/ci_cd.yml) | tests → build → deploy |

## Full tree (key paths)

```
MLops/
├── notebooks/                 # 01_eda, 02_model_selection
├── models/                    # model.py (resnet18|cnn), train.py, utils.py
├── app/
│   ├── api/main.py            # FastAPI serving + Prometheus metrics + drift gauges
│   └── webapp/streamlit_app.py
├── airflow/dags/              # data_pipeline, retrain_pipeline (drift-aware)
├── retrain/                   # drift-gated automated retraining
├── monitoring/
│   ├── prometheus.yml
│   ├── grafana/dashboards/    # dandelion, airflow, drift
│   └── drift/                 # PSI engine + Evidently report + sample report
├── k8s/minikube-manifest.yaml
├── tests/                     # test_utils, test_model, test_api, test_drift
├── docs/                      # MODEL_CARD, PROJECT_STRUCTURE, LOOM_SCRIPT, SPECIFICATION
├── deliverables/              # cahier_des_charges.docx, presentation.pptx
├── Dockerfile · docker-compose.yml
├── requirements.txt · requirements-dev.txt · requirements-monitoring.txt
└── .github/workflows/ci_cd.yml
```
