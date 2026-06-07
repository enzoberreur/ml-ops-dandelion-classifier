"""Dependency-light drift detection engine (NumPy + pandas only).

Drift is quantified per feature with the Population Stability Index (PSI), a
standard, model-agnostic metric in production ML monitoring. The conventional
reading is:

* PSI < 0.1  -> no significant shift
* 0.1 <= PSI < 0.2 -> moderate shift (watch)
* PSI >= 0.2 -> significant shift (act / retrain)

A dataset is flagged as drifted when the share of significantly-drifted features
exceeds ``dataset_drift_share`` (default 50%). This engine is the trigger used by
the automated retraining pipeline; the Evidently report layer is purely for
human-facing visualisation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

DEFAULT_PSI_THRESHOLD = 0.2
DEFAULT_DATASET_DRIFT_SHARE = 0.5


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute the PSI between a reference and a current distribution.

    Bucket edges are derived from the reference quantiles so the metric is robust
    to skewed features. ``epsilon`` guards against empty buckets / log(0).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # constant reference feature -> no measurable drift
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


@dataclass
class DriftResult:
    """Outcome of a drift evaluation across all monitored features."""

    feature_psi: Dict[str, float] = field(default_factory=dict)
    drifted_features: List[str] = field(default_factory=list)
    n_features: int = 0
    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    dataset_drift_share: float = DEFAULT_DATASET_DRIFT_SHARE

    @property
    def share_drifted(self) -> float:
        if self.n_features == 0:
            return 0.0
        return len(self.drifted_features) / self.n_features

    @property
    def dataset_drift(self) -> bool:
        """True when enough features drifted to warrant retraining."""
        return self.share_drifted >= self.dataset_drift_share

    def to_dict(self) -> Dict[str, object]:
        return {
            "dataset_drift": self.dataset_drift,
            "share_drifted": round(self.share_drifted, 4),
            "n_features": self.n_features,
            "n_drifted": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "psi_threshold": self.psi_threshold,
            "dataset_drift_share": self.dataset_drift_share,
            "feature_psi": {k: round(v, 4) for k, v in self.feature_psi.items()},
        }


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    dataset_drift_share: float = DEFAULT_DATASET_DRIFT_SHARE,
) -> DriftResult:
    """Compute per-feature PSI and aggregate into a :class:`DriftResult`.

    Only numeric columns present in both frames are evaluated.
    """
    numeric_cols = [
        c
        for c in reference.columns
        if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]
    feature_psi: Dict[str, float] = {}
    drifted: List[str] = []
    for col in numeric_cols:
        psi = population_stability_index(reference[col].to_numpy(), current[col].to_numpy())
        feature_psi[col] = psi
        if psi >= psi_threshold:
            drifted.append(col)

    return DriftResult(
        feature_psi=feature_psi,
        drifted_features=drifted,
        n_features=len(numeric_cols),
        psi_threshold=psi_threshold,
        dataset_drift_share=dataset_drift_share,
    )
