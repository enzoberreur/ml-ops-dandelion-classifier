"""Tests for the drift monitoring engine (NumPy/pandas only, no torch needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from monitoring.drift.detect import compute_drift, population_stability_index
from monitoring.drift.features import FEATURE_COLUMNS, image_features
from monitoring.drift.report import build_report


def test_psi_zero_for_identical_distribution() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    assert population_stability_index(x, x.copy()) < 1e-6


def test_psi_increases_with_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    small = population_stability_index(ref, rng.normal(0.2, 1, 2000))
    large = population_stability_index(ref, rng.normal(3.0, 1, 2000))
    assert large > small
    assert large >= 0.2  # a 3-sigma shift is clearly significant


def test_compute_drift_flags_shifted_feature() -> None:
    rng = np.random.default_rng(1)
    n = 500
    reference = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(0, 1, n)})
    current = pd.DataFrame({"stable": rng.normal(0, 1, n), "shifted": rng.normal(4, 1, n)})
    result = compute_drift(reference, current)
    assert "shifted" in result.drifted_features
    assert "stable" not in result.drifted_features
    assert result.n_features == 2


def test_image_features_returns_expected_columns() -> None:
    img = Image.new("RGB", (40, 30), color=(120, 200, 60))
    feats = image_features(img)
    assert set(feats) == set(FEATURE_COLUMNS)
    assert feats["aspect_ratio"] == 40 / 30


def test_build_report_writes_html_and_metrics(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    reference = pd.DataFrame({f: rng.normal(0, 1, 300) for f in ("a", "b", "c")})
    current = pd.DataFrame({f: rng.normal(3, 1, 300) for f in ("a", "b", "c")})
    html_path = tmp_path / "report.html"
    metrics_path = tmp_path / "metrics.json"
    result = build_report(reference, current, html_path, metrics_path)
    assert html_path.exists() and html_path.stat().st_size > 0
    assert metrics_path.exists()
    assert result.dataset_drift is True
