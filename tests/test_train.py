"""Test the evaluation metrics path (accuracy, F1, precision, recall, confusion matrix)."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image

from models.model import build_model
from models.train import evaluate
from models.utils import CLASS_NAMES, DataConfig, create_dataloaders


@pytest.fixture()
def tmp_dataset(tmp_path: Path) -> Path:
    for ci, class_name in enumerate(CLASS_NAMES):
        d = tmp_path / class_name
        d.mkdir(parents=True)
        for idx in range(4):
            shade = 30 if ci == 0 else 200
            Image.new("RGB", (32, 32), color=(shade, shade, shade)).save(d / f"{idx}.jpg")
    return tmp_path


def test_evaluate_returns_full_metrics(tmp_dataset: Path) -> None:
    cfg = DataConfig(data_dir=tmp_dataset, image_size=32, batch_size=2, val_split=0.5)
    _, val_loader, classes = create_dataloaders(cfg)
    model = build_model(arch="cnn", num_classes=len(classes), pretrained=False)
    metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert set(metrics) == {"loss", "accuracy", "f1", "precision", "recall", "confusion_matrix"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert len(cm) == len(classes) and len(cm[0]) == len(classes)
