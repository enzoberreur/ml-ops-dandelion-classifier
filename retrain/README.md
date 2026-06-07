# Automated Retraining

Closes the *Automated retraining* criterion. Retraining is triggered two ways,
both reusing the **same drift definition** as production monitoring (the PSI
engine in [`/monitoring/drift`](../monitoring/drift/)):

| Mode | Trigger | Where |
|------|---------|-------|
| **Scheduled refresh** | Monthly cadence | `dandelion_retrain_pipeline` Airflow DAG (`@monthly`) |
| **Drift-triggered** | Incoming data drifted vs the model's reference set | `retrain/retrain.py` drift gate |

## The retraining loop

```
sync latest processed data (MinIO)
        │
        ▼
data-drift check  ──►  drift_report.html / drift_metrics.json / drift.prom
        │
        ▼
retrain?  ── scheduled refresh → yes
          └ drift-gated        → only if dataset drift detected
        │
        ▼
train (models.train) → log to MLflow → push best_model.pt to MinIO
        │
        ▼
refresh reference baseline  → next run compares against this dataset
```

The API picks up the new checkpoint from MinIO on its next start/reload, so a
retrain propagates to serving without code changes.

## Usage

```bash
# Drift-gated: retrain ONLY if the current set drifted vs the reference
python -m retrain.retrain --data-dir data/processed --reference-dir data/baseline

# Forced scheduled refresh (what the monthly DAG does)
python -m retrain.retrain --data-dir data/processed --force

# Forward extra args to the trainer (e.g. switch architecture / epochs)
python -m retrain.retrain --data-dir data/processed --force -- --arch resnet18 --epochs 8
```

Exit codes: `0` handled (trained or correctly skipped) · `2` error (surfaced to
the orchestrator).

## Why gate on drift?

Retraining on an unchanged distribution wastes compute and risks silently
swapping a validated model for a noisier one. Gating on drift means the model is
refreshed **when the world changed**, and the decision is auditable: every gated
run writes the drift report that justified (or vetoed) the retrain.
