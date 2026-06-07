"""CLI: compare a reference image set against the current set and report drift.

Used both interactively and as the decision step of the automated retraining
pipeline. Exit code is ``1`` when dataset drift is detected so an orchestrator
(Airflow branch, CI job) can react to it; ``0`` otherwise.

Examples
--------
    # Local directories
    python -m monitoring.drift.run_drift_check \
        --reference data/processed --current data/incoming \
        --out-dir monitoring/drift/reports

    # The generated artifacts:
    #   reports/drift_report.html     (Evidently or fallback)
    #   reports/drift_metrics.json    (machine-readable summary)
    #   reports/drift.prom            (Prometheus textfile)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from monitoring.drift.features import features_from_directory
from monitoring.drift.report import build_report, write_prometheus_textfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
LOG = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image data drift check")
    parser.add_argument("--reference", type=Path, required=True, help="Reference image directory")
    parser.add_argument("--current", type=Path, required=True, help="Current/incoming image directory")
    parser.add_argument("--out-dir", type=Path, default=Path("monitoring/drift/reports"))
    parser.add_argument("--limit", type=int, default=None, help="Cap images per set (speed)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference = features_from_directory(args.reference, limit=args.limit)
    current = features_from_directory(args.current, limit=args.limit)
    if reference.empty or current.empty:
        LOG.error("No images found (reference=%d, current=%d rows)", len(reference), len(current))
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = build_report(
        reference,
        current,
        out_html=args.out_dir / "drift_report.html",
        out_metrics=args.out_dir / "drift_metrics.json",
    )
    write_prometheus_textfile(result, args.out_dir / "drift.prom")

    LOG.info("Drift summary: %s", json.dumps(result.to_dict()))
    LOG.info("Report written to %s", args.out_dir / "drift_report.html")
    return 1 if result.dataset_drift else 0


if __name__ == "__main__":
    sys.exit(main())
