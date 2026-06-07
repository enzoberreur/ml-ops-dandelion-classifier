# Drift Monitoring

Production monitoring for the dandelion classifier, covering **data drift** (have
the incoming photos shifted away from the training distribution?). This is the
component required by the *Production monitoring* criterion and feeds the
**drift-triggered retraining** logic in [`/retrain`](../../retrain/).

## Design

Two layers, on purpose:

| Layer | Module | Dependencies | Role |
|-------|--------|--------------|------|
| Detection | `detect.py` | NumPy / pandas | PSI engine - the retraining trigger. Always runs. |
| Reporting | `report.py` | `evidently` (optional) | Rich HTML report; falls back to a self-rendered HTML if Evidently is absent. |

Images can't be fed to tabular drift tools directly, so `features.py` reduces each
image to interpretable statistics (brightness, contrast, per-channel intensity,
saturation, edge density, aspect ratio). Drift in these features is a practical
proxy for covariate shift (lighting, season, camera).

### Drift metric - Population Stability Index (PSI)

| PSI | Reading |
|-----|---------|
| < 0.1 | no significant shift |
| 0.1 - 0.2 | moderate shift (watch) |
| ≥ 0.2 | significant shift (act / retrain) |

A dataset is flagged drifted when **≥ 50%** of features cross the PSI threshold.

## Run it

```bash
# Reproducible sample (synthetic reference vs shifted current) - no pipeline needed
python -m monitoring.drift.generate_sample_report
# -> monitoring/drift/reports/sample_drift_report.html (+ .json, + .prom)

# Real check on two image directories
python -m monitoring.drift.run_drift_check \
  --reference data/processed --current data/incoming \
  --out-dir monitoring/drift/reports
# exit code 1 == dataset drift detected (consumable by Airflow / CI)
```

## Outputs

| Artifact | Purpose |
|----------|---------|
| `drift_report.html` | Human-facing report (Evidently `DataDriftPreset`, or fallback). |
| `drift_metrics.json` | Machine-readable summary (per-feature PSI, drifted share, decision). |
| `drift.prom` | Prometheus textfile (`dandelion_dataset_drift`, `dandelion_drift_share`, `dandelion_feature_psi`). |

## Wiring to Grafana / Prometheus

`drift.prom` is emitted in Prometheus *textfile-collector* format. Mount the
output directory into a `node_exporter --collector.textfile.directory` (or scrape
it) to add `dandelion_dataset_drift` and `dandelion_drift_share` gauges to the
existing dashboards and alert when drift = 1.

A committed sample (`reports/sample_drift_*.{html,json,prom}`) is included so the
deliverable is demoable without running the full data pipeline.
