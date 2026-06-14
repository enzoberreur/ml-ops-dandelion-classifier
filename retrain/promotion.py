"""Model promotion gate: only ship a retrained model if it is good enough.

A retrained candidate must clear two bars before it replaces the model the API
serves from MinIO:

* **Quality floor** - macro F1 of at least ``min_f1`` (default 0.85).
* **No regression** - macro F1 not worse than the currently promoted model by
  more than a small tolerance (0.01), absorbing validation-split noise.

Metric keys match the ``metrics_summary.json`` written by ``models/train.py``
(``val_f1`` is the macro F1). The promoted model's summary is stored next to
the checkpoint in MinIO as ``latest_metrics.json`` so the next retrain can
compare against it.
"""
from __future__ import annotations

import json
import logging
from typing import Optional, Tuple

LOG = logging.getLogger(__name__)

# Key of the macro F1 score in the metrics summary written by models/train.py.
F1_KEY = "val_f1"
# Allowed macro F1 drop vs the current model before a candidate is blocked.
REGRESSION_TOLERANCE = 0.01


def should_promote(
    candidate_metrics: dict,
    current_metrics: Optional[dict],
    min_f1: float = 0.85,
) -> Tuple[bool, str]:
    """Decide whether a candidate model should replace the current one.

    Pure function: returns ``(decision, reason)`` so callers can log an
    auditable explanation alongside the upload (or the refusal).
    """
    candidate_f1 = float(candidate_metrics.get(F1_KEY, 0.0))
    if candidate_f1 < min_f1:
        return False, f"candidate {F1_KEY}={candidate_f1:.4f} is below the quality floor min_f1={min_f1:.2f}"
    if current_metrics is None:
        return True, f"no current model metrics; candidate {F1_KEY}={candidate_f1:.4f} clears min_f1={min_f1:.2f}"
    current_f1 = float(current_metrics.get(F1_KEY, 0.0))
    if candidate_f1 >= current_f1 - REGRESSION_TOLERANCE:
        return True, (
            f"candidate {F1_KEY}={candidate_f1:.4f} is within tolerance of current "
            f"{F1_KEY}={current_f1:.4f} (tolerance={REGRESSION_TOLERANCE})"
        )
    return False, (
        f"candidate {F1_KEY}={candidate_f1:.4f} regressed beyond tolerance vs current "
        f"{F1_KEY}={current_f1:.4f} (tolerance={REGRESSION_TOLERANCE})"
    )


def metrics_object_for(model_object: str) -> str:
    """Return the ``latest_metrics.json`` object key sitting next to the checkpoint."""
    prefix, _, _ = model_object.rpartition("/")
    return f"{prefix}/latest_metrics.json" if prefix else "latest_metrics.json"


def fetch_current_metrics(client, bucket: str, object_name: str) -> Optional[dict]:
    """Load the promoted model's metrics summary from MinIO, or ``None`` if absent.

    Any failure (missing bucket/object, network error, bad JSON) is treated as
    "no current metrics", which routes the caller onto the first-promotion path.
    """
    try:
        response = client.get_object(bucket, object_name)
        try:
            return json.loads(response.read().decode("utf-8"))
        finally:
            response.close()
            response.release_conn()
    except Exception as exc:  # noqa: BLE001 - absence of metrics is not an error
        LOG.info("No current metrics at s3://%s/%s (%s); treating as first promotion", bucket, object_name, exc)
        return None
