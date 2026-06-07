"""FastAPI service exposing the dandelion classifier."""
from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from models.model import DEFAULT_ARCH, build_model
from models.utils import CLASS_NAMES, get_inference_transform, get_minio_client

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")

app = FastAPI(title="Dandelion Classifier API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_LOCAL_PATH = Path(os.environ.get("MODEL_LOCAL_PATH", "/tmp/model/best_model.pt"))
MINIO_MODEL_BUCKET = os.environ.get("MINIO_MODEL_BUCKET", "dandelion-models")
MINIO_MODEL_PATH = os.environ.get("MINIO_MODEL_PATH", "models/latest/best_model.pt")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 128))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: torch.nn.Module | None = None
class_names = CLASS_NAMES
model_arch = DEFAULT_ARCH
transform = get_inference_transform(IMAGE_SIZE)
model_ready = False

PREDICTION_COUNTER = Counter("predictions_total", "Number of predictions served", ["result"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency for prediction endpoint")

# Data-drift gauges, refreshed from the JSON the drift pipeline writes. This lets
# the existing Prometheus scrape of /metrics surface drift with no extra service.
DRIFT_METRICS_PATH = Path(os.environ.get("DRIFT_METRICS_PATH", "monitoring/drift/reports/drift_metrics.json"))
DRIFT_DATASET_GAUGE = Gauge("dandelion_dataset_drift", "1 if dataset drift detected, else 0")
DRIFT_SHARE_GAUGE = Gauge("dandelion_drift_share", "Share of features with significant drift")
DRIFT_PSI_GAUGE = Gauge("dandelion_feature_psi", "Per-feature Population Stability Index", ["feature"])


def refresh_drift_gauges(path: Path = DRIFT_METRICS_PATH) -> bool:
    """Load the latest drift summary into the Prometheus gauges. Best-effort."""
    try:
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        DRIFT_DATASET_GAUGE.set(1 if data.get("dataset_drift") else 0)
        DRIFT_SHARE_GAUGE.set(float(data.get("share_drifted", 0.0)))
        for feature, psi in (data.get("feature_psi") or {}).items():
            DRIFT_PSI_GAUGE.labels(feature=feature).set(float(psi))
        return True
    except Exception as exc:  # never let metrics scraping fail
        LOG.debug("Drift gauge refresh skipped: %s", exc)
        return False


def _download_model() -> None:
    client = get_minio_client()
    MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    client.fget_object(MINIO_MODEL_BUCKET, MINIO_MODEL_PATH, str(MODEL_LOCAL_PATH))
    LOG.info("Downloaded model from MinIO at %s", MINIO_MODEL_PATH)


def _load_model() -> None:
    global model, class_names, model_arch, model_ready  # noqa: PLW0603 - module level cache
    if not MODEL_LOCAL_PATH.exists():
        _download_model()
    checkpoint = torch.load(MODEL_LOCAL_PATH, map_location=device)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    # Checkpoints trained before the multi-arch refactor have no "arch" key and
    # were always the custom CNN, so default to "cnn" for backward compatibility.
    model_arch = checkpoint.get("arch", "cnn")
    # Rebuild the backbone without downloading ImageNet weights: the trained
    # weights from the checkpoint are loaded straight after.
    model_instance = build_model(arch=model_arch, num_classes=len(class_names), pretrained=False)
    model_instance.load_state_dict(checkpoint["model_state_dict"])
    model_instance.eval()
    model_instance.to(device)
    model = model_instance
    LOG.info("Model loaded (arch=%s) with classes: %s", model_arch, class_names)
    model_ready = True


@app.on_event("startup")
async def startup_event() -> None:
    try:
        _load_model()
    except Exception as exc:  # pragma: no cover - startup failure should be visible in logs
        LOG.warning("Model not available at startup: %s", exc)
        # The service remains up; subsequent prediction calls will fail until model is loaded.


@app.get("/health")
def health() -> Dict[str, Any]:
    status = "ready" if model_ready else "initializing"
    return {"status": status, "classes": class_names}


@app.get("/version")
def version() -> Dict[str, Any]:
    """Expose the served model metadata (architecture, classes, source object)."""
    return {
        "api_version": app.version,
        "model_arch": model_arch,
        "model_ready": model_ready,
        "classes": class_names,
        "model_source": f"s3://{MINIO_MODEL_BUCKET}/{MINIO_MODEL_PATH}",
        "image_size": IMAGE_SIZE,
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:  # pragma: no cover - guard user input
        LOG.error("Invalid image input: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    start = time.perf_counter()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    duration = time.perf_counter() - start
    PREDICTION_LATENCY.observe(duration)

    prediction = class_names[predicted_idx.item()]
    PREDICTION_COUNTER.labels(result=prediction).inc()
    return {
        "prediction": prediction,
        "confidence": round(float(confidence.item()), 4),
        "class_probabilities": {
            class_names[i]: round(float(probabilities[i].item()), 4) for i in range(len(class_names))
        },
    }


@app.get("/metrics")
def metrics() -> Response:
    refresh_drift_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
