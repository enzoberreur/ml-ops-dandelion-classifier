"""Automated, drift-gated retraining entrypoint.

Two trigger modes are supported, matching real production practice:

* **Scheduled refresh** - run unconditionally on a cadence (the monthly Airflow
  DAG). Use ``--force``.
* **Drift-triggered** - only retrain when the incoming data has drifted away from
  the reference distribution the current model was trained on. Provide a
  ``--reference-dir`` (or a MinIO feature baseline) and the drift gate decides.

The drift gate reuses the same PSI engine that powers production monitoring, so
"the dashboard says drift" and "the pipeline retrained" share one definition of
drift. Exit code ``0`` = handled (trained or correctly skipped), ``2`` = error.

    # Drift-gated (only retrains if the current set drifted vs the reference)
    python -m retrain.retrain --data-dir data/processed --reference-dir data/baseline

    # Forced scheduled refresh
    python -m retrain.retrain --data-dir data/processed --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from monitoring.drift.detect import DriftResult
from monitoring.drift.features import features_from_directory
from monitoring.drift.report import build_report, write_prometheus_textfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
LOG = logging.getLogger(__name__)


def evaluate_drift(reference_dir: Path, current_dir: Path, out_dir: Path) -> Optional[DriftResult]:
    """Compute drift between two image directories and persist the artifacts.

    Returns ``None`` when either side has no usable images (caller decides what
    to do - typically: retrain, since we cannot prove the model is still valid).
    """
    reference = features_from_directory(reference_dir)
    current = features_from_directory(current_dir)
    if reference.empty or current.empty:
        LOG.warning("Drift check skipped: reference=%d current=%d images", len(reference), len(current))
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_report(
        reference,
        current,
        out_html=out_dir / "drift_report.html",
        out_metrics=out_dir / "drift_metrics.json",
    )
    write_prometheus_textfile(result, out_dir / "drift.prom")
    LOG.info("Drift result: %s", json.dumps(result.to_dict()))
    return result


def retrain(data_dir: Path, extra_args: Optional[List[str]] = None) -> None:
    """Invoke the training entrypoint on ``data_dir``."""
    from models.train import train as train_model  # imported lazily (torch heavy)

    argv = ["--data-dir", str(data_dir)]
    if extra_args:
        argv += extra_args
    LOG.info("Launching training: %s", argv)
    train_model(argv)


def gated_retrain(
    data_dir: Path,
    reference_dir: Optional[Path] = None,
    out_dir: Path = Path("monitoring/drift/reports"),
    force: bool = False,
    extra_args: Optional[List[str]] = None,
) -> dict:
    """Decide whether to retrain, then act. Returns a structured decision record."""
    decision = {"retrained": False, "reason": "", "drift": None}

    if force or reference_dir is None:
        decision["reason"] = "forced/scheduled refresh" if force else "no reference provided"
        retrain(data_dir, extra_args)
        decision["retrained"] = True
        return decision

    result = evaluate_drift(reference_dir, data_dir, out_dir)
    if result is None:
        decision["reason"] = "insufficient data for drift check -> retrain to be safe"
        retrain(data_dir, extra_args)
        decision["retrained"] = True
        return decision

    decision["drift"] = result.to_dict()
    if result.dataset_drift:
        decision["reason"] = f"dataset drift detected (share={result.share_drifted:.0%})"
        retrain(data_dir, extra_args)
        decision["retrained"] = True
    else:
        decision["reason"] = f"no significant drift (share={result.share_drifted:.0%}) -> skip"
    return decision


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drift-gated automated retraining")
    parser.add_argument("--data-dir", type=Path, required=True, help="Dataset to (potentially) train on")
    parser.add_argument("--reference-dir", type=Path, default=None, help="Reference image set for the drift gate")
    parser.add_argument("--out-dir", type=Path, default=Path("monitoring/drift/reports"))
    parser.add_argument("--force", action="store_true", help="Skip the drift gate and always retrain")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="Extra args forwarded to models.train")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    extra = args.train_args[1:] if args.train_args and args.train_args[0] == "--" else args.train_args
    try:
        decision = gated_retrain(
            data_dir=args.data_dir,
            reference_dir=args.reference_dir,
            out_dir=args.out_dir,
            force=args.force,
            extra_args=extra or None,
        )
    except Exception as exc:  # surface as nonzero exit for the orchestrator
        LOG.exception("Retraining failed: %s", exc)
        return 2
    LOG.info("Decision: %s", json.dumps(decision.get("reason")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
