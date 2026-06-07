"""Turn an image dataset into a tabular feature frame for drift analysis.

Image drift cannot be fed directly to tabular drift tools, so each image is
reduced to a handful of cheap, interpretable statistics (per-channel intensity,
brightness, contrast, saturation, edge density). Shifts in these features are a
practical proxy for covariate shift in the incoming photos (lighting, season,
camera). The same frame format is consumed by both the PSI engine and Evidently.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

FEATURE_COLUMNS = [
    "brightness",
    "contrast",
    "red_mean",
    "green_mean",
    "blue_mean",
    "saturation_mean",
    "edge_density",
    "aspect_ratio",
]


def image_features(image: Image.Image) -> Dict[str, float]:
    """Extract interpretable scalar features from a single PIL image."""
    rgb = image.convert("RGB")
    stat = ImageStat.Stat(rgb)
    r_mean, g_mean, b_mean = stat.mean
    brightness = float(np.mean(stat.mean))
    contrast = float(np.mean(stat.stddev))

    hsv = np.asarray(rgb.convert("HSV"), dtype=float)
    saturation_mean = float(hsv[:, :, 1].mean())

    gray = np.asarray(rgb.convert("L"), dtype=float)
    gy, gx = np.gradient(gray)
    edge_density = float(np.mean(np.hypot(gx, gy)))

    width, height = rgb.size
    aspect_ratio = float(width / height) if height else 0.0

    return {
        "brightness": brightness,
        "contrast": contrast,
        "red_mean": float(r_mean),
        "green_mean": float(g_mean),
        "blue_mean": float(b_mean),
        "saturation_mean": saturation_mean,
        "edge_density": edge_density,
        "aspect_ratio": aspect_ratio,
    }


def features_from_directory(image_dir: Path, limit: int | None = None) -> pd.DataFrame:
    """Build a feature frame from every ``.jpg``/``.png`` under ``image_dir``."""
    image_dir = Path(image_dir)
    paths: List[Path] = sorted(
        p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit is not None:
        paths = paths[:limit]
    records: List[Dict[str, float]] = []
    for path in paths:
        try:
            with Image.open(path) as img:
                records.append(image_features(img))
        except Exception:  # skip unreadable files, keep the batch going
            continue
    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)
