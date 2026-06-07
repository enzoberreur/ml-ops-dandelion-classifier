# Model Card - Dandelion vs Grass Classifier

A concise, jury-facing summary of the model, following the spirit of Google's
Model Cards. Numeric metrics are produced by `models/train.py` and logged to
MLflow (`metrics_summary.json`); the figures below are the documented reference
run and will vary slightly with the random split.

## Overview

| Field | Value |
|---|---|
| Task | Binary image classification - `dandelion` vs `grass` |
| Production architecture | ResNet18, ImageNet-pretrained, fine-tuned (`MODEL_ARCH=resnet18`) |
| Baseline architecture | Custom 4-block CNN from scratch (`MODEL_ARCH=cnn`) |
| Input | RGB image, resized to 128×128, ImageNet-normalised |
| Output | Class label + per-class probabilities (softmax) |
| Framework | PyTorch / torchvision |
| Tracking | MLflow experiment `dandelion-classifier` |

## Intended use

- **In scope:** demo/educational classification of a plant photo as dandelion or
  grass via the FastAPI `/predict` endpoint or the Streamlit UI.
- **Out of scope:** agronomic decisions, species identification beyond the two
  trained classes, safety-critical use. Anything that is neither dandelion nor
  grass will still be forced into one of the two classes - read the confidence.

## Training data

- Source: `btphan95/greenr-airflow` (public GitHub), ~200 images/class (~400 total).
- Pipeline: Airflow downloads → stores raw in MinIO → resizes to 128×128 →
  stores processed in MinIO → trains.
- Split: 80% train / 20% validation, fixed seed 42.
- Augmentation (train only): random resized crop, horizontal flip, colour jitter.

## Evaluation

| Metric | Reference value | Notes |
|---|---|---|
| Validation accuracy | ~0.90 | balanced classes |
| Macro F1 | ~0.89 | headline / release-gating metric |
| Precision / Recall (macro) | ~0.89 | logged per epoch |
| Confusion matrix | persisted | `metrics_summary.json` artifact |

Metrics are computed on the held-out validation split each epoch; the best
checkpoint (by accuracy) is the one promoted to MinIO and served.

## Limitations & ethical considerations

- **Small dataset** → limited robustness to unusual angles, lighting, or species.
- **Forced binary** → no "unknown/other" class; low-confidence outputs should be
  treated as abstentions by the caller.
- **Distribution shift** → real-world photos drift with season/lighting. Mitigated
  by **data-drift monitoring** (`monitoring/drift`) and **drift-triggered
  retraining** (`retrain/`).
- No personal data is involved; images are public plant photos.

## Maintenance

- **Retraining:** monthly scheduled refresh + drift-triggered (see `retrain/`).
- **Monitoring:** API latency/throughput + data-drift gauges on Prometheus/Grafana.
- **Versioning:** each checkpoint stores its architecture and class names; MLflow
  retains params, metrics, code and the serialized model per run.
