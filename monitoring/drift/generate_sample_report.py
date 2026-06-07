"""Generate a committed sample drift report from synthetic feature data.

This makes the monitoring deliverable reproducible and demoable without needing
the full data pipeline: it fabricates a reference distribution and a *shifted*
current distribution (simulating a lighting/season change in incoming photos),
then writes the same artifacts the real pipeline produces. Deterministic seed so
the committed sample is stable.

    python -m monitoring.drift.generate_sample_report
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift.features import FEATURE_COLUMNS
from monitoring.drift.report import build_report, write_prometheus_textfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
LOG = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent / "reports"


def _synth(n: int, rng: np.random.Generator, shift: float = 0.0) -> pd.DataFrame:
    """Synthesize an image-feature frame; ``shift`` injects covariate drift."""
    data = {
        "brightness": rng.normal(120 + 35 * shift, 18, n),
        "contrast": rng.normal(48 + 10 * shift, 8, n),
        "red_mean": rng.normal(110 + 25 * shift, 20, n),
        "green_mean": rng.normal(135, 20, n),
        "blue_mean": rng.normal(80 - 15 * shift, 18, n),
        "saturation_mean": rng.normal(150 + 40 * shift, 25, n),
        "edge_density": rng.normal(28, 6, n),
        "aspect_ratio": rng.normal(1.0, 0.05, n),
    }
    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def main() -> None:
    rng = np.random.default_rng(42)
    reference = _synth(400, rng, shift=0.0)
    current = _synth(400, rng, shift=1.0)  # brighter, warmer, more saturated -> drift

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Canonical filenames: this committed sample doubles as the API's initial
    # drift data and is overwritten by the first real pipeline/CLI run.
    result = build_report(
        reference,
        current,
        out_html=OUT_DIR / "drift_report.html",
        out_metrics=OUT_DIR / "drift_metrics.json",
    )
    write_prometheus_textfile(result, OUT_DIR / "drift.prom")
    LOG.info("Sample drift report written to %s", OUT_DIR / "drift_report.html")
    LOG.info("Dataset drift detected: %s (share=%.0f%%)", result.dataset_drift, result.share_drifted * 100)


if __name__ == "__main__":
    main()
