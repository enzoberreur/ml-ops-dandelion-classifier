"""Drift reporting layer.

Produces a human-facing HTML report. When the optional ``evidently`` package is
installed (it is pinned in ``requirements.txt`` and present in the Docker/CI
image), a full Evidently ``DataDriftPreset`` report is rendered. Otherwise a
self-contained HTML fallback is generated from the PSI engine so the pipeline
never hard-fails on a missing optional dependency.
"""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from monitoring.drift.detect import DriftResult, compute_drift


def _evidently_available() -> bool:
    try:
        import evidently  # noqa: F401
    except Exception:  # pragma: no cover - depends on environment
        return False
    return True


def evidently_html_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> bool:
    """Render an Evidently DataDriftPreset report. Returns True on success."""
    try:  # pragma: no cover - exercised only when evidently is installed
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        out_html.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(out_html))
        return True
    except Exception:
        return False


def _fallback_html(result: DriftResult, out_html: Path) -> None:
    """Self-contained HTML report rendered from the PSI engine (no dependencies)."""
    rows = "".join(
        f"<tr class='{'drift' if psi >= result.psi_threshold else 'ok'}'>"
        f"<td>{html.escape(feature)}</td>"
        f"<td>{psi:.4f}</td>"
        f"<td>{'DRIFT' if psi >= result.psi_threshold else 'stable'}</td></tr>"
        for feature, psi in sorted(result.feature_psi.items(), key=lambda kv: kv[1], reverse=True)
    )
    banner_color = "#c0392b" if result.dataset_drift else "#27ae60"
    banner_text = "DATASET DRIFT DETECTED" if result.dataset_drift else "No significant dataset drift"
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Drift report - dandelion classifier</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; color: #222; }}
 h1 {{ font-size: 1.4rem; }}
 .banner {{ background: {banner_color}; color: #fff; padding: 12px 16px; border-radius: 8px; font-weight: 600; }}
 .meta {{ color: #555; margin: 12px 0; }}
 table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
 th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
 th {{ background: #f5f6f8; }}
 tr.drift td {{ background: #fdecea; }}
 .note {{ color: #777; font-size: 0.85rem; margin-top: 1.5rem; }}
</style></head><body>
<h1>Data drift report - dandelion classifier</h1>
<div class="banner">{banner_text}</div>
<p class="meta">{result.n_features} features monitored · {len(result.drifted_features)} drifted ·
 share drifted {result.share_drifted:.0%} (threshold {result.dataset_drift_share:.0%}) ·
 PSI threshold {result.psi_threshold}</p>
<table><thead><tr><th>Feature</th><th>PSI</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note">Fallback report (Evidently not installed in this environment).
 PSI &lt; 0.1 = no shift · 0.1-0.2 = moderate · &ge; 0.2 = significant.</p>
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(doc, encoding="utf-8")


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_html: Path,
    out_metrics: Optional[Path] = None,
) -> DriftResult:
    """Compute drift, write an HTML report (Evidently or fallback) and a JSON summary."""
    result = compute_drift(reference, current)
    used_evidently = evidently_html_report(reference, current, out_html)
    if not used_evidently:
        _fallback_html(result, out_html)

    if out_metrics is not None:
        payload = result.to_dict()
        payload["report_engine"] = "evidently" if used_evidently else "fallback"
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        out_metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return result


def write_prometheus_textfile(result: DriftResult, out_prom: Path) -> None:
    """Emit drift gauges in Prometheus textfile-collector format.

    Mount the output directory into a node_exporter ``--collector.textfile`` path
    (or scrape it) to surface drift on the existing Grafana stack.
    """
    lines = [
        "# HELP dandelion_dataset_drift 1 if dataset drift detected, else 0",
        "# TYPE dandelion_dataset_drift gauge",
        f"dandelion_dataset_drift {1 if result.dataset_drift else 0}",
        "# HELP dandelion_drift_share Share of features with significant drift",
        "# TYPE dandelion_drift_share gauge",
        f"dandelion_drift_share {result.share_drifted:.4f}",
        "# HELP dandelion_feature_psi Per-feature Population Stability Index",
        "# TYPE dandelion_feature_psi gauge",
    ]
    lines += [f'dandelion_feature_psi{{feature="{f}"}} {psi:.4f}' for f, psi in result.feature_psi.items()]
    out_prom.parent.mkdir(parents=True, exist_ok=True)
    out_prom.write_text("\n".join(lines) + "\n", encoding="utf-8")
