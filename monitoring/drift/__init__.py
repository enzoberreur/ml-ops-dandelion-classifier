"""Data & prediction drift monitoring for the dandelion classifier.

The package is intentionally split into two layers:

* :mod:`monitoring.drift.detect` - a dependency-light drift engine (NumPy/pandas
  only) that powers the automated retraining trigger. It always runs, even in a
  minimal container.
* :mod:`monitoring.drift.report` - a richer reporting layer that produces an
  Evidently HTML report when the optional ``evidently`` dependency is installed,
  and falls back to a self-rendered HTML table otherwise.
"""
from monitoring.drift.detect import DriftResult, compute_drift, population_stability_index

__all__ = ["DriftResult", "compute_drift", "population_stability_index"]
