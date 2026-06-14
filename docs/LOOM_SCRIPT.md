# Loom Demo Script - Bloc 4 (AI Solution / MLOps)

**Goal:** in one take, show everything the jury grades: the specification, the ML
model, the serving API, the CI/CD pipeline, production monitoring with drift, and
automated retraining.

**Required flow:** Prediction API -> CI/CD -> Monitoring (drift, metrics) -> Retraining.

**Target length:** 5 minutes. Record at 1920x1080, browser zoom ~110%.

**Project:** GreenGuard, a self-maintaining dandelion-vs-grass image classifier
(ResNet18 transfer learning), industrialized end to end.

---

## Pre-flight (off camera, so the recording stays smooth)

1. `docker compose up -d`; `docker compose ps` all healthy.
2. Trigger the data pipeline once and confirm one training run exists so a model is
   served and MLflow has a run to open.
3. **Reconcile the metric you will narrate:** open MLflow and note the actual
   `val_f1` / `val_accuracy` on the served run, and say that number on camera (do
   not quote a figure that disagrees with `artifacts/metrics_summary.json`).
4. Have `test_dandelion.jpg` and `test_grass.jpg` ready to upload.
5. Tabs ready: Streamlit `http://localhost:8501`, API docs `http://localhost:8000/docs`,
   MLflow `http://localhost:5500`, GitHub **Actions**, Grafana `http://localhost:3000`,
   Airflow `http://localhost:8080`.

---

## 0:00 - 0:30 - Intro and architecture  *(criterion: Specifications and business relevance, 15%)*

**SAY:** "This is an end-to-end MLOps solution: a dandelion-vs-grass classifier,
industrialized. Airflow orchestration, MLflow tracking, MinIO storage, a FastAPI
serving layer, CI/CD on GitHub Actions, and production monitoring with data-drift
detection and automated retraining. The business case is chemical-free weeding,
where mistaking a crop for a weed has a real cost."

**SHOW:** the README architecture diagram and the service table.

---

## 0:30 - 1:00 - Specification  *(criterion: Specifications and business relevance, 15%)*

**SAY:** "It starts from a written specification."

**SHOW:** `docs/SPECIFICATION.md` (or the rendered
`deliverables/cahier_des_charges.docx`): objectives and KPIs, functional and
non-functional requirements, and the table mapping each business outcome to one ML
metric and one live monitoring signal, plus the definition of done.

---

## 1:00 - 1:45 - Prediction API, the product  *(criterion: Serving API quality, 15%)*

**SHOW:** Streamlit `http://localhost:8501`. Upload `test_dandelion.jpg` -> show the
prediction and confidence. Upload `test_grass.jpg` -> the other class.

**SHOW:** API docs `http://localhost:8000/docs`.
- `GET /version` -> point out `model_arch: resnet18`, the class names, the model
  source.
- `POST /predict` on an image -> the JSON with prediction plus per-class
  probabilities.
- Mention `GET /health` and `GET /metrics`, and that the API returns 400 on a
  non-image and 503 while the model is loading (covered by `tests/test_api.py`).

---

## 1:45 - 2:30 - ML model and tracking  *(criterion: ML model quality, 25%)*

**SHOW:** MLflow `http://localhost:5500` -> the latest `dandelion-classifier` run.
- Params: `arch=resnet18`, epochs, learning rate.
- Metrics: `val_accuracy`, `val_f1`, precision/recall, confusion matrix.

**SAY:** "ResNet18 transfer learning on about 400 images. Read the actual val_f1
here - every run logs its code, params, metrics and the model artifact, so it is
fully reproducible: fixed seed, pinned dependencies." Then show `docs/MODEL_CARD.md`
(data, evaluation, limitations, maintenance) and mention the 22 tests in `tests/`.

---

## 2:30 - 3:15 - CI/CD  *(criterion: CI/CD pipeline, 20%)*

**SHOW:** the repo **Actions** tab -> the latest green `CI/CD` run. Walk the three
chained jobs:
- **tests**: pytest (model, API, drift) with coverage to Codecov.
- **build**: Docker image built and pushed to GHCR, SHA-tagged.
- **deploy**: `kubeconform` validates the Kubernetes manifests against the schema.

**SAY:** "Every push to main runs the tests, builds and publishes the image, and
validates the deployment manifests." Open `.github/workflows/ci_cd.yml` briefly.

---

## 3:15 - 4:00 - Monitoring: metrics and drift  *(criterion: Production monitoring, 10%)*

**SHOW:** Grafana `http://localhost:3000`:
- **Dandelion Classifier** dashboard: total predictions and p90 latency moving as
  you hit the API.
- **Dandelion Data Drift** dashboard: dataset drift status, share of features
  drifted, per-feature PSI bars.

**SHOW:** `monitoring/drift/reports/drift_report.html`.
**SAY:** "Drift is PSI over eight interpretable image features, reported by
Evidently with a dependency-free fallback. The same drift gauges are exposed on the
API's /metrics endpoint, so the dashboard and the retrain decision use one
definition of drift."

---

## 4:00 - 4:40 - Automated retraining  *(criterion: Automated retraining, 10%)*

**SHOW:** Airflow `http://localhost:8080` -> `dandelion_retrain_pipeline`. Show the
graph: `sync -> check_drift -> train -> refresh_baseline`.

**SAY:** "Retraining runs monthly and on-demand when drift is detected, using the
same drift definition as the dashboard. A candidate is promoted only if it clears a
0.85 F1 floor and does not regress against the live model - that gate is in
`retrain/promotion.py` with seven unit tests covering every branch." Trigger the DAG
to show it running (optional).

---

## 4:40 - 5:00 - Wrap  *(criterion: Presentation and Q&A, 5%)*

**SAY:** "Specification, model, serving API, CI/CD, monitoring, and retraining: a
closed MLOps loop where the model watches its own inputs and replaces itself only
when the replacement is provably better. Everything is in the repo." Show the repo
root tree briefly and mention `docs/DEFENSE.md` (Q&A pack) and `docs/EVALUATION.md`.

Stop the stack after recording: `docker compose down`.

---

## Rubric coverage map

| Criterion | Weight | Shown at |
|---|---|---|
| Specifications and business relevance | 15% | 0:00, 0:30 |
| ML model quality | 25% | 1:45 |
| Serving API quality | 15% | 1:00 |
| CI/CD pipeline | 20% | 2:30 |
| Automated retraining | 10% | 4:00 |
| Production monitoring | 10% | 3:15 |
| Presentation and Q&A | 5% | 0:00 + 4:40 |

## Shot checklist
- [ ] Architecture diagram + specification (SPECIFICATION.md / cahier)
- [ ] Streamlit prediction (both classes)
- [ ] `/version` + `/predict` JSON (+ /health, /metrics)
- [ ] MLflow run with the real val_f1 + model card
- [ ] GitHub Actions green run (tests -> build -> deploy)
- [ ] Grafana classifier dashboard + drift dashboard
- [ ] `drift_report.html`
- [ ] Airflow retrain DAG graph + promotion gate
- [ ] Repo tree on the wrap

## Paste the Loom URL into
- `README.md` (top, a "Demo video" line)
- `docs/EVALUATION.md`
- The closing slide of `deliverables/presentation.pptx`

## One reminder before you record
Reconcile the metrics story first: the narrated F1 must match
`artifacts/metrics_summary.json`. If that file still shows a perfect 1.0, either
retrain to a realistic number or add a held-out test split, so what you say matches
what the jury can open.
