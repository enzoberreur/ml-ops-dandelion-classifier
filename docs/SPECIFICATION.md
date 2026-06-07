# Cahier des Charges - AI Solution Specification

**Project:** GreenGuard - Vision and MLOps Platform for Autonomous Chemical-Free Weeding Robots
**Bloc 4 - Artificial Intelligence Solutions**
**Authors:** Enzo Berreur, Elea Nizam, Jean-Baptiste Brun, Elisa Leclerc
**Version:** 1.0

---

## 1. Context & Business Problem

Chemical weed control is being phased out. Regulations such as France's Loi Labbe
ban synthetic herbicides in public green spaces and private gardens, organic
farming forbids them, and grounds-care faces a chronic labour shortage. The
emerging answer is the **autonomous weeding robot**: a machine that drives over a
field, lawn or roadside and removes weeds mechanically, with no chemicals.

Such a robot is only as good as its eyes. For every patch it passes it must
decide, in real time, whether it sees a **weed** (here, a dandelion, to be
removed) or the **crop / turf** (here, grass, to be left untouched). Cut the wrong
thing and it damages the very lawn or crop it is meant to protect.

**GreenGuard** is that perception brain and the MLOps platform behind it: a binary
image classifier (dandelion vs grass) served at the edge, wrapped in the
industrialised pipeline that keeps it accurate as it meets new sites, seasons and
lighting. This document specifies the solution as a production-grade MLOps system:
not just a model, but a serving API, automated deployment, monitoring, and
self-maintenance through drift-triggered retraining.

**Why MLOps is not optional here.** A fleet of robots works across many fields and
seasons, so the images it sees drift continuously. Without automatic drift
detection and retraining the model silently degrades, and a degraded model means a
robot that uproots the crop or leaves weeds standing. Monitoring and retraining are
therefore the reliability backbone of the product, not add-ons.

### Indicative business case

The driver is regulatory and economic, not a marginal optimisation:

- **Compliance:** where herbicides are banned (for example France's Loi Labbe for
  public green spaces), chemical weeding is simply not an option; a chemical-free
  robot is the compliant one.
- **Labour:** autonomous operation removes the recurring cost of manual weeding,
  the main expense in pesticide-free grounds care.
- **Asset protection:** accurate perception avoids destroying the crop or turf the
  robot exists to protect.

Exact figures are deployment-specific. The point for this specification is that the
model's accuracy, and its sustained accuracy over time, translate directly into
chemicals avoided, labour saved and crop protected, which is why the MLOps loop
(monitoring plus retraining) is core to the value rather than peripheral.

## 2. Objectives & Success Criteria

| Objective | Success criterion (KPI) |
|---|---|
| Accurate classification | Macro-F1 ≥ 0.85 on the validation set |
| Responsive serving | p90 prediction latency < 500 ms (CPU) |
| Reliable availability | API health endpoint green; graceful startup without a model |
| Reproducible ML | Every training run tracked (params, metrics, model) in MLflow |
| Self-maintaining | Drift detected automatically; retraining triggered on drift or monthly |
| Observable | Live dashboards for throughput, latency and data drift |

### From business value to monitored metrics

Monitoring is wired to business outcomes, not just technical health:

| Business outcome | ML metric | Live monitoring signal |
|---|---|---|
| Crop / turf safety (never cut the crop) | per-class precision and recall, confusion matrix | prediction mix, drift (PSI), retrain events |
| Effective weeding (do not miss weeds) | macro-F1, recall on the weed class | release-gate macro-F1; drift flag in Grafana |
| Real-time operation on a moving robot | inference latency | p90 prediction_latency_seconds (Prometheus) |
| Sustained accuracy across sites and seasons | drift magnitude, retrain cadence | dataset_drift flag, drift_share, retrain runs |
| Low maintenance cost | automation coverage | green CI/CD, automated drift-triggered retraining |
| Herbicide avoided and uptime | model availability | predictions_total, /health readiness |

## 3. Scope

**In scope:** the ML model (training + evaluation), a REST serving API, a demo
web UI, orchestration of data and retraining pipelines, CI/CD automation,
production monitoring (operational + drift), and infrastructure-as-code for a
Kubernetes deployment.

**Out of scope:** the robot hardware and its mechanical weeding actuator, the
on-robot edge-runtime integration, and a multi-species weed taxonomy (future work).
The model is intentionally binary (weed vs crop) as the proof-of-concept perception
core.

## 4. Stakeholders

| Stakeholder | Interest / role |
|---|---|
| Product owner | Business value, KPIs, scope and priorities |
| ML engineer | Model design, training, evaluation, drift strategy |
| MLOps / platform engineer | CI/CD, deployment, monitoring, reliability |
| Robotics OEM | Embeds the perception module; needs a stable, updatable model API |
| Robot operator / grounds team | Reliable weed-vs-crop calls so the robot removes only weeds |
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

- **CI/CD:** on every push to `main` - run tests, build and publish the container
  image to the registry, and validate the Kubernetes deployment manifests
  (schema-checked, offline). The same manifests deploy to a local Minikube cluster.
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

- All tests green in CI; image built and published; Kubernetes manifests validated.
  The stack deploys to a local Minikube cluster.
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
