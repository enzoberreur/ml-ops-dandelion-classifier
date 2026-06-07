"""Streamlit UI allowing API or local MinIO inference for the classifier."""
from __future__ import annotations

import io
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List

import requests
import streamlit as st
import torch
from minio.error import S3Error
from PIL import Image

from models.model import build_model
from models.utils import CLASS_NAMES, get_inference_transform, get_minio_client

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MINIO_MODEL_BUCKET = os.environ.get("MINIO_MODEL_BUCKET", "dandelion-models")
MINIO_MODEL_PREFIX = os.environ.get("MINIO_MODEL_PREFIX", "models/")
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", 128))
DEVICE = torch.device("cpu")
TRANSFORM = get_inference_transform(IMAGE_SIZE)

st.set_page_config(page_title="Dandelion vs Grass", page_icon="🌼")
st.title("🌼 Dandelion or Grass?")
st.write(
    "Testez vos modèles PyTorch en direct depuis MinIO ou interrogez l'API "
    "déployée pour obtenir des prédictions."
)


@st.cache_data(show_spinner=False)
def list_available_models() -> List[str]:
    """Retourne la liste des checkpoints stockés sur MinIO."""
    client = get_minio_client()
    try:
        objects = client.list_objects(MINIO_MODEL_BUCKET, prefix=MINIO_MODEL_PREFIX, recursive=True)
        return sorted(
            obj.object_name
            for obj in objects
            if obj.object_name.lower().endswith(".pt")
        )
    except S3Error as exc:  # pragma: no cover - dépend du réseau
        st.warning(f"Impossible de lister les modèles dans MinIO : {exc}")
        return []


@st.cache_resource(show_spinner=False)
def load_model_from_minio(object_name: str) -> torch.nn.Module:
    """Télécharge un checkpoint depuis MinIO et le charge en mémoire."""
    client = get_minio_client()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        local_path = Path(tmp_file.name)
    try:
        client.fget_object(MINIO_MODEL_BUCKET, object_name, str(local_path))
        checkpoint = torch.load(local_path, map_location=DEVICE)
    finally:
        with suppress(FileNotFoundError):
            local_path.unlink(missing_ok=True)

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    arch = checkpoint.get("arch", "cnn")
    model = build_model(arch=arch, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_names = class_names  # type: ignore[attr-defined]
    model.eval()
    model.to(DEVICE)
    return model


def render_api_mode(uploaded_file: Any) -> None:
    """Envoie l'image à l'API FastAPI pour obtenir une prédiction."""
    if uploaded_file and st.button("Prédire via l'API"):
        st.info("Envoi de l'image à l'API…")
        try:
            response = requests.post(
                f"{API_URL}/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        io.BytesIO(uploaded_file.getvalue()),
                        uploaded_file.type or "image/jpeg",
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            payload: Dict[str, Any] = response.json()
        except requests.RequestException as exc:
            st.error(f"Appel API impossible : {exc}")
            return
        st.success(f"Prédiction : {payload['prediction']} (confiance {payload['confidence']:.2f})")
        st.json(payload.get("class_probabilities", {}))
    else:
        st.caption("Configurez l'URL de l'API dans la barre latérale si nécessaire.")


def render_local_mode(uploaded_file: Any) -> None:
    """Effectue l'inférence localement en téléchargeant un modèle depuis MinIO."""
    model_choices = list_available_models()
    if not model_choices:
        st.warning("Aucun modèle .pt trouvé dans le bucket MinIO.")
        return

    selected_model = st.selectbox("Choisissez un modèle", model_choices)

    if uploaded_file and st.button("Prédire avec le modèle local"):
        with st.spinner("Téléchargement du modèle…"):
            try:
                model = load_model_from_minio(selected_model)
            except S3Error as exc:
                st.error(f"Impossible de télécharger le modèle : {exc}")
                return

        image = Image.open(uploaded_file).convert("RGB")
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, dim=0)

        class_names = getattr(model, "class_names", CLASS_NAMES)
        prediction = class_names[predicted_idx.item()]
        st.success(f"Prédiction : {prediction} (confiance {confidence.item():.2f})")
        st.json(
            {
                class_names[i]: round(float(probabilities[i].item()), 4)
                for i in range(len(class_names))
            }
        )


with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("URL de l'API FastAPI", value=API_URL)
    if api_url != API_URL:
        API_URL = api_url
    mode = st.radio("Mode d'inférence", ["API distante", "Local (MinIO)"])
    st.caption("Le mode local télécharge directement un checkpoint et exécute l'inférence ici.")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Image téléchargée", use_column_width=True)
else:
    st.info("Ajoutez une image pour lancer une prédiction.")

if uploaded_file:
    if mode == "API distante":
        render_api_mode(uploaded_file)
    else:
        render_local_mode(uploaded_file)
