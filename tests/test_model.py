"""Tests for the model factory (both architectures)."""
from __future__ import annotations

import pytest
import torch

from models.model import DEFAULT_ARCH, build_model


def test_default_arch_is_resnet18() -> None:
    assert DEFAULT_ARCH == "resnet18"


@pytest.mark.parametrize("arch", ["cnn", "resnet18"])
def test_forward_pass_shape(arch: str) -> None:
    # pretrained=False keeps the test offline (no weight download).
    model = build_model(arch=arch, num_classes=2, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 2)


def test_unknown_arch_raises() -> None:
    with pytest.raises(ValueError):
        build_model(arch="not-a-real-arch")
