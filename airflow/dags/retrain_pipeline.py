"""Airflow DAG: drift-aware automated retraining.

Each run: sync the latest processed dataset from MinIO, run a data-drift check
against the reference baseline (the dataset the live model was last trained on),
publish the drift artifacts/metrics, retrain on the monthly cadence, then refresh
the baseline. The drift report is what the on-demand, *drift-triggered* path in
``retrain/retrain.py`` consumes - same drift definition, one source of truth.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

from models.train import train as train_model
from models.utils import get_minio_client, pull_directory_from_minio
from retrain.retrain import evaluate_drift

DATA_BASE_PATH = Path(Variable.get("data_base_path", "/opt/airflow/data"))
APP_DIR = Path(Variable.get("app_dir", "/opt/airflow/app"))
PROCESSED_DATA_DIR = DATA_BASE_PATH / "processed"
REFERENCE_DATA_DIR = DATA_BASE_PATH / "reference"
# Write to the shared repo path so the API container (which mounts the same repo)
# surfaces the drift gauges on /metrics for Prometheus/Grafana.
DRIFT_REPORT_DIR = APP_DIR / "monitoring" / "drift" / "reports"
MINIO_DATA_BUCKET = Variable.get("minio_data_bucket", "dandelion-data")


def sync_processed_from_minio() -> None:
    client = get_minio_client()
    pull_directory_from_minio(client, MINIO_DATA_BUCKET, "processed", PROCESSED_DATA_DIR)


def check_drift() -> None:
    """Compare the freshly-synced dataset against the reference baseline.

    Writes ``drift_report.html`` / ``drift_metrics.json`` / ``drift.prom`` so the
    shift is visible on the monitoring stack. Defensive by design: it never fails
    the DAG (the first run simply seeds the baseline).
    """
    has_reference = REFERENCE_DATA_DIR.exists() and any(REFERENCE_DATA_DIR.rglob("*.jpg"))
    if not has_reference:
        DRIFT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (DRIFT_REPORT_DIR / "status.txt").write_text(
            "No reference baseline yet; it will be seeded after this run.\n", encoding="utf-8"
        )
        return
    result = evaluate_drift(REFERENCE_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_REPORT_DIR)
    if result is not None and result.dataset_drift:
        # Surfaced in logs and on the Grafana drift panel via drift.prom.
        print(f"[drift] dataset drift detected: share={result.share_drifted:.0%}")


def run_training() -> None:
    train_model(["--data-dir", str(PROCESSED_DATA_DIR)])


def refresh_baseline() -> None:
    """Promote the just-trained-on dataset to the reference baseline."""
    if not PROCESSED_DATA_DIR.exists():
        return
    if REFERENCE_DATA_DIR.exists():
        shutil.rmtree(REFERENCE_DATA_DIR)
    shutil.copytree(PROCESSED_DATA_DIR, REFERENCE_DATA_DIR)


def build_dag() -> DAG:
    default_args = {
        "owner": "mlops",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="dandelion_retrain_pipeline",
        default_args=default_args,
        schedule="@monthly",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["dandelion", "mlops", "retrain", "drift"],
    ) as dag:
        sync = PythonOperator(task_id="sync_processed_data", python_callable=sync_processed_from_minio)
        drift = PythonOperator(task_id="check_drift", python_callable=check_drift)
        train = PythonOperator(task_id="train_model", python_callable=run_training)
        baseline = PythonOperator(task_id="refresh_baseline", python_callable=refresh_baseline)

        sync >> drift >> train >> baseline

    return dag


globals()["dandelion_retrain_pipeline"] = build_dag()
