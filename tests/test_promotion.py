"""Tests for the model promotion gate (retrain/promotion.py)."""
from __future__ import annotations

from retrain.promotion import metrics_object_for, should_promote


def test_first_promotion_without_current_metrics() -> None:
    promote, reason = should_promote({"val_f1": 0.90}, None)
    assert promote is True
    assert "no current model metrics" in reason


def test_first_promotion_blocked_below_min_f1() -> None:
    promote, reason = should_promote({"val_f1": 0.80}, None)
    assert promote is False
    assert "quality floor" in reason


def test_promote_on_improvement() -> None:
    promote, reason = should_promote({"val_f1": 0.93}, {"val_f1": 0.90})
    assert promote is True
    assert "within tolerance" in reason


def test_promote_within_tolerance_of_current() -> None:
    # 0.895 vs current 0.90 is inside the 0.01 tolerance band.
    promote, _ = should_promote({"val_f1": 0.895}, {"val_f1": 0.90})
    assert promote is True


def test_block_on_regression_beyond_tolerance() -> None:
    promote, reason = should_promote({"val_f1": 0.86}, {"val_f1": 0.95})
    assert promote is False
    assert "regressed beyond tolerance" in reason


def test_block_below_min_f1_even_if_current_is_worse() -> None:
    promote, reason = should_promote({"val_f1": 0.70}, {"val_f1": 0.60})
    assert promote is False
    assert "quality floor" in reason


def test_metrics_object_sits_next_to_checkpoint() -> None:
    assert metrics_object_for("models/latest/best_model.pt") == "models/latest/latest_metrics.json"
    assert metrics_object_for("best_model.pt") == "latest_metrics.json"
