"""Model definitions for the dandelion vs grass classifier.

Two architectures are supported and selectable at train/serve time via the
``MODEL_ARCH`` environment variable (or the ``--arch`` CLI flag):

* ``resnet18`` (default) - transfer learning from an ImageNet-pretrained
  ResNet18 with a fresh 2-class head. This is the production choice: on a small
  dataset (~400 images) a pretrained backbone converges faster and generalises
  better than training from scratch.
* ``cnn`` - a small custom CNN trained from scratch, kept as a lightweight
  baseline that requires no model download (useful for fully offline runs).

The selected architecture name is persisted inside the model checkpoint so the
serving layer can rebuild the exact same network without guessing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

DEFAULT_ARCH = "resnet18"
SUPPORTED_ARCHS = ("resnet18", "cnn")


class DandelionClassifier(nn.Module):
    """Small CNN baseline tailored for 128x128 RGB inputs (no pretrained weights)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        return self.classifier(x)


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    """ResNet18 backbone with a fresh classification head (transfer learning)."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Factory returning the requested architecture.

    Args:
        arch: ``"resnet18"`` (transfer learning) or ``"cnn"`` (from-scratch baseline).
        num_classes: number of output classes.
        pretrained: load ImageNet weights for the ResNet backbone. Ignored for ``cnn``.
            Set to ``False`` for fully offline environments (no weight download).
    """
    arch = (arch or DEFAULT_ARCH).lower()
    if arch == "resnet18":
        return _build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if arch == "cnn":
        return DandelionClassifier(num_classes=num_classes)
    raise ValueError(f"Unsupported architecture '{arch}'. Choose one of {SUPPORTED_ARCHS}.")
