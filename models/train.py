"""Training entrypoint for the dandelion vs grass classifier."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from models.model import DEFAULT_ARCH, build_model
from models.utils import (
    CLASS_NAMES,
    DataConfig,
    create_dataloaders,
    get_device,
    get_minio_client,
    seed_everything,
    upload_file_to_minio,
)

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dandelion vs grass classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Path to processed dataset root")
    parser.add_argument("--image-size", type=int, default=int(os.environ.get("IMAGE_SIZE", 128)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("TRAINING_BATCH_SIZE", 32)))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("TRAINING_EPOCHS", 5)))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", 1e-3)))
    parser.add_argument("--arch", type=str, default=os.environ.get("MODEL_ARCH", DEFAULT_ARCH), choices=["resnet18", "cnn"], help="Model architecture")
    parser.add_argument("--pretrained", type=lambda v: str(v).lower() not in {"0", "false", "no"}, default=os.environ.get("MODEL_PRETRAINED", "true").lower() not in {"0", "false", "no"}, help="Use ImageNet weights for the ResNet backbone")
    parser.add_argument("--experiment-name", type=str, default="dandelion-classifier")
    parser.add_argument("--run-name", type=str, default="training-run")
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/best_model.pt"))
    parser.add_argument("--minio-model-bucket", type=str, default=os.environ.get("MINIO_MODEL_BUCKET", "dandelion-models"))
    parser.add_argument("--minio-model-path", type=str, default="models/latest/best_model.pt")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", 0)))
    return parser.parse_args(argv)


def evaluate(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, object]:
    """Run a full validation pass and return loss + classification metrics.

    Returns accuracy, macro F1 / precision / recall and the confusion matrix so
    the training loop can log them to MLflow (matching what the docs promise).
    """
    model.eval()
    loss_total = 0.0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_total += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            total += targets.size(0)

    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_targets)) / max(1, total)
    f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    precision = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    recall = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    cm = confusion_matrix(all_targets, all_preds).tolist()
    return {
        "loss": loss_total / max(1, total),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def train(argv: list[str] | None = None) -> None:
    import mlflow  # lazy: training needs it, but evaluate()/CLI parsing do not

    args = parse_args(argv)
    seed_everything()
    device = get_device()
    LOG.info("Using device: %s", device)

    data_config = DataConfig(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader, val_loader, class_names = create_dataloaders(data_config)
    LOG.info("Classes discovered: %s", class_names)

    model = build_model(arch=args.arch, num_classes=len(class_names), pretrained=args.pretrained).to(device)
    LOG.info("Model architecture: %s (pretrained=%s)", args.arch, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(args.experiment_name)

    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"))
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", "minioadmin"))

    best_accuracy = 0.0
    best_metrics: Dict[str, object] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "confusion_matrix": []}
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(
            {
                "arch": args.arch,
                "pretrained": args.pretrained,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "image_size": args.image_size,
            }
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            scheduler.step()
            train_loss = epoch_loss / max(1, total)
            train_accuracy = correct / max(1, total)
            val_metrics = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": float(val_metrics["loss"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                },
                step=epoch,
            )

            LOG.info(
                "Epoch %d/%d - train_loss: %.4f train_acc: %.4f val_loss: %.4f val_acc: %.4f val_f1: %.4f",
                epoch,
                args.epochs,
                train_loss,
                train_accuracy,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["f1"],
            )

            if val_metrics["accuracy"] >= best_accuracy:
                best_accuracy = float(val_metrics["accuracy"])
                best_metrics = val_metrics
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "arch": args.arch,
                        "image_size": args.image_size,
                    },
                    args.model_output,
                )
                mlflow.log_artifact(args.model_output, artifact_path="checkpoints")

        # Persist the final evaluation summary (incl. confusion matrix) for the model card.
        metrics_summary = {
            "best_val_accuracy": best_accuracy,
            "val_f1": float(best_metrics["f1"]),
            "val_precision": float(best_metrics["precision"]),
            "val_recall": float(best_metrics["recall"]),
            "confusion_matrix": best_metrics["confusion_matrix"],
            "class_names": class_names,
            "arch": args.arch,
        }
        # Local copy next to the checkpoint (consumed by notebook 02 and the model card).
        (args.model_output.parent / "metrics_summary.json").write_text(
            json.dumps(metrics_summary, indent=2), encoding="utf-8"
        )
        mlflow.log_metric("best_val_accuracy", best_accuracy)
        mlflow.log_metric("best_val_f1", float(best_metrics["f1"]))
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="code")
        mlflow.log_text(json.dumps({"class_names": class_names}), artifact_file="class_names.json")
        mlflow.log_text(json.dumps(metrics_summary, indent=2), artifact_file="metrics_summary.json")
        mlflow.pytorch.log_model(model, artifact_path="model")

    # Promotion gate: only replace the served model when the candidate clears the
    # quality floor and does not regress vs the currently promoted model.
    from retrain.promotion import fetch_current_metrics, metrics_object_for, should_promote

    client = get_minio_client()
    metrics_object = metrics_object_for(args.minio_model_path)
    current_metrics = fetch_current_metrics(client, args.minio_model_bucket, metrics_object)
    promote, reason = should_promote(metrics_summary, current_metrics)
    if promote:
        upload_file_to_minio(client, args.minio_model_bucket, args.minio_model_path, args.model_output)
        upload_file_to_minio(
            client,
            args.minio_model_bucket,
            metrics_object,
            args.model_output.parent / "metrics_summary.json",
            content_type="application/json",
        )
        LOG.info("Promotion gate: PROMOTE - %s", reason)
        LOG.info("Model uploaded to MinIO bucket=%s object=%s", args.minio_model_bucket, args.minio_model_path)
    else:
        LOG.warning("Promotion gate: BLOCK - %s", reason)
        LOG.warning("Candidate kept local at %s; MinIO model left unchanged", args.model_output)


if __name__ == "__main__":
    train()
