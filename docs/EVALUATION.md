# Evaluation criteria coverage - Bloc 4 (Solutions d'IA)

This table maps every jury grading criterion to exactly where it is demonstrated,
across the three graded artifacts: the slide deck, this repository, and the Loom
demo. Use it during the defense so each point is easy to verify.

| Criterion | Weight | Where it is proven |
|-----------|:------:|--------------------|
| ML model | 25% | `models/` (model factory: ResNet18 transfer-learning default + CNN baseline, architecture saved in the checkpoint). `notebooks/01_eda` + `notebooks/02_model_selection` (data exploration + model choice rationale). `docs/MODEL_CARD.md`. Metrics logged: F1 / precision / recall + confusion matrix + `metrics_summary.json`. |
| CI/CD | 20% | `.github/workflows/ci_cd.yml`: tests + coverage -> Docker build and push to GHCR -> Kubernetes manifest validation (`kubeconform`). `Dockerfile`, `docker-compose.yml`, `k8s/minikube-manifest.yaml`. |
| Specs and business relevance | 15% | `deliverables/cahier_des_charges.docx` + `docs/SPECIFICATION.md`: the vision/MLOps platform for autonomous chemical-free weeding robots (drivers: herbicide bans, labour shortage), with a business-outcome -> ML-metric -> monitoring-signal table and an indicative business case. |
| Serving API | 15% | `app/api/` FastAPI service: `/predict`, `/health`, `/version`, `/metrics`. `app/webapp/` Streamlit UI for interactive prediction. |
| Automated retraining | 10% | `retrain/` (drift-gated retraining logic) + `airflow/dags/dandelion_retrain_pipeline` (triggers retraining when drift crosses the threshold). |
| Production monitoring | 10% | `monitoring/` Prometheus + Grafana, plus data-drift detection (`monitoring/drift/`: PSI engine + Evidently report + `drift.json` Grafana dashboard) exposed as gauges on `/metrics`. |
| Presentation | 5% | `deliverables/presentation.pptx` + Streamlit demo + 5-minute oral. See `docs/DEFENSE.md` for the Q&A. |

**Reading tip for the jury:** the highest weights are ML model (25%) and CI/CD (20%).
The model work (selection rationale, per-class metrics, model card) is in `models/`,
`notebooks/`, and `docs/MODEL_CARD.md`; the full closed loop (CI/CD, retraining,
drift monitoring) is what makes this an MLOps solution rather than just a model.
