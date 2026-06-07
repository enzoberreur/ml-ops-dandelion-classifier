# Cahier des Charges - AI Solution Specification

**Project:** GreenGuard - Automated Weed Detection for Smart Gardening
**Bloc 4 - Artificial Intelligence Solutions**
**Authors:** Enzo Berreur, Elea Nizam, Jean-Baptiste Brun, Elisa Leclerc
**Version:** 1.0

---

## 1. Context & Business Problem

Home gardeners and grounds-keepers routinely apply herbicide to entire lawns to
remove a handful of weeds - wasteful, costly, and environmentally harmful.
**GreenGuard** is a smart-gardening assistant: the user photographs a patch of
lawn from a mobile app and the system flags whether it contains **dandelions**
(a weed to spot-treat) or is **grass** (leave alone). This enables targeted,
spot-treatment instead of blanket spraying.

The AI component is a binary image classifier (dandelion vs grass). This document
specifies that component as a production-grade, industrialised MLOps solution:
not just a model, but a serving API, automated deployment, monitoring, and
self-maintenance through retraining.

## 2. Objectives & Success Criteria

| Objective | Success criterion (KPI) |
|---|---|
| Accurate classification | Macro-F1 ≥ 0.85 on the validation set |
| Responsive serving | p90 prediction latency < 500 ms (CPU) |
| Reliable availability | API health endpoint green; graceful startup without a model |
| Reproducible ML | Every training run tracked (params, metrics, model) in MLflow |
| Self-maintaining | Drift detected automatically; retraining triggered on drift or monthly |
| Observable | Live dashboards for throughput, latency and data drift |

## 3. Scope

**In scope:** the ML model (training + evaluation), a REST serving API, a demo
web UI, orchestration of data and retraining pipelines, CI/CD automation,
production monitoring (operational + drift), and infrastructure-as-code for a
Kubernetes deployment.

**Out of scope:** the mobile application front-end, user account management, and
multi-species weed taxonomy (future work). The model is intentionally binary.

## 4. Stakeholders

| Stakeholder | Interest / role |
|---|---|
| Product owner | Business value, KPIs, scope and priorities |
| ML engineer | Model design, training, evaluation, drift strategy |
| MLOps / platform engineer | CI/CD, deployment, monitoring, reliability |
| End user (gardener) | Fast, trustworthy predictions from a photo |
| RNCP jury | Evaluates the solution against the competency framework |

## 5. Functional Requirements

1. The system shall ingest labelled images for two classes (dandelion, grass) and
   prepare them for training (resize, normalise).
2. The system shall train an image classifier and record metrics and the model
   artifact for every run.
3. The serving API shall expose `POST /predict` returning the predicted class and
   per-class probabilities for an uploaded image.
4. The serving API shall expose `GET /health`, `GET /version` (model metadata)
   and `GET /metrics` (Prometheus).
5. The system shall provide a web UI to upload an image and view the prediction.
6. The system shall detect data drift between the incoming and reference image
   distributions and produce a drift report.
7. The system shall retrain the model on a schedule and on detected drift, and
   promote the new model to serving.

## 6. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | p90 inference latency < 500 ms on CPU; training < ~10 min on CPU |
| Availability | API starts even if no model is present yet (returns 503 on predict until ready) |
| Scalability | Stateless API, horizontally scalable behind a Kubernetes Service |
| Security | Secrets via environment/K8s Secrets; no credentials in code; least-privilege CI token |
| Maintainability | Tested code, typed where practical, documented; one-command local stack |
| Portability | Runs identically via Docker Compose (local) and Minikube (cluster) |
| Cost | Lightweight model (ResNet18) and open-source stack; no managed-service lock-in |
| Reproducibility | Fixed seeds, pinned dependencies, MLflow run tracking |

## 7. Data Requirements

- **Source:** public image dataset (`btphan95/greenr-airflow`), ~200 images/class.
- **Storage:** raw and processed images in MinIO (S3-compatible), versioned by
  pipeline run.
- **Preparation:** convert to RGB, resize/fit to 128×128, ImageNet normalisation.
- **Governance:** no personal data; public plant imagery only. Lineage is traced
  via Airflow run IDs and MLflow tags.

## 8. ML Requirements & Acceptance Thresholds

- **Architecture:** ResNet18 transfer learning (production) with a custom-CNN
  baseline available for fully offline runs.
- **Metrics:** accuracy, macro F1, precision, recall, confusion matrix.
- **Release gate:** a candidate model is promotable only if macro-F1 ≥ 0.85 on the
  validation split.
- **Tracking:** params, metrics, code and the serialized model logged to MLflow.

## 9. MLOps Requirements

- **CI/CD:** on every push to `main` - run tests, build the container image, push
  to the registry, and deploy to Kubernetes (Minikube) with rollout verification.
- **Retraining:** monthly scheduled refresh plus drift-triggered retraining, using
  the same drift definition as monitoring.
- **Monitoring:** operational metrics (prediction count, p90 latency) and data
  drift (per-feature PSI, dataset-drift flag) on Prometheus + Grafana; a
  human-readable drift report (Evidently, with a dependency-free fallback).

## 10. Architecture & Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Orchestration | Apache Airflow | Declarative, observable pipelines for data + retraining |
| Model | PyTorch / torchvision ResNet18 | Transfer learning ideal for a small dataset |
| Tracking | MLflow | Standard experiment tracking + model registry semantics |
| Storage | MinIO (S3 API) | Reproducible artifact/data store, cloud-portable |
| Serving | FastAPI + Streamlit | Fast async API + simple demo UI |
| Packaging | Docker / Docker Compose / Kubernetes | Identical local and cluster runtime |
| CI/CD | GitHub Actions | Native to the repository host, free for public repos |
| Monitoring | Prometheus + Grafana + Evidently | Operational + drift observability |

A full architecture diagram and component walkthrough are in the slide deck and
`README.md`.

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Small dataset → overfitting | Lower real-world accuracy | Transfer learning + augmentation + validation gating |
| Data/concept drift | Silent accuracy decay | Drift monitoring + drift-triggered retraining |
| Model unavailable at startup | Failed predictions | API stays up, returns 503 until model loads |
| Dependency conflicts (Airflow vs Evidently) | Broken stack | Evidently isolated as optional; graceful fallback |
| Forced binary output on unknown input | Misleading prediction | Expose confidence; document out-of-scope use |

## 12. Acceptance Criteria (Definition of Done)

- All tests green in CI; image built and deployed to the cluster.
- `POST /predict` returns a class + probabilities for a valid image; `400` on a
  non-image; `/health` and `/version` respond.
- A training run logs metrics to MLflow and promotes a checkpoint meeting the
  release gate.
- A drift report and Grafana drift panels render; the retrain pipeline runs.
- README, model card and this specification are complete and accurate.

## 13. Glossary

- **Drift:** change in the input data distribution vs the model's training data.
- **PSI:** Population Stability Index, a per-feature drift metric.
- **Transfer learning:** fine-tuning a pretrained network on a new task.
- **IaC:** Infrastructure as Code (here: Kubernetes manifests, Compose).
