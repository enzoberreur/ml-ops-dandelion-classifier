"""API tests using FastAPI's TestClient with an in-memory model (no MinIO)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture()
def client(monkeypatch):
    """Build the app with a tiny in-memory model, bypassing MinIO download."""
    from app.api import main
    from models.model import build_model

    model = build_model(arch="cnn", num_classes=2, pretrained=False)
    model.eval()
    monkeypatch.setattr(main, "model", model)
    monkeypatch.setattr(main, "model_ready", True)
    monkeypatch.setattr(main, "model_arch", "cnn")
    return TestClient(main.app)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(80, 160, 60)).save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_version_reports_arch(client) -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_arch"] == "cnn"
    assert "classes" in body


def test_predict_returns_known_class(client) -> None:
    resp = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"dandelion", "grass"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["class_probabilities"]) == {"dandelion", "grass"}


def test_predict_rejects_non_image(client) -> None:
    resp = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert resp.status_code == 400


def test_metrics_endpoint(client) -> None:
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"predictions_total" in resp.content
