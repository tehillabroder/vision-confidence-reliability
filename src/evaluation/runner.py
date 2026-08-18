"""Shared helpers for degradation evaluation."""

from typing import Iterable, Optional
import torch
import torch.nn as nn
from src.metrics.reliability import calibration_bins

def build_experiment_conditions(degradations: list[str], severity_levels: list[int]) -> list[tuple[str, int]]:
    conditions = [("none", 0)]
    for degradation in degradations:
        for severity in severity_levels:
            conditions.append((degradation, severity))
    return conditions

def build_calibration_rows(rows: list[dict], metadata: dict) -> list[dict]:
    if not rows:
        raise ValueError("Cannot build calibration rows from empty predictions.")
    if "ece_bins" not in metadata:
        raise ValueError("Calibration metadata must include ece_bins.")

    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    condition_rows = calibration_bins(correct, confidences, n_bins=metadata["ece_bins"])

    # keep the columns in whatever order the caller passes them in so existing CSV files and table layouts don't break
    for row in condition_rows:
        row.update(metadata)

    return condition_rows

@torch.no_grad()
def collect_prediction_rows(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    dataset_name: str,
    model_name: str,
    degradation: str,
    severity: int,
    seed: int,
    max_eval_batches: Optional[int]
) -> list[dict]:
    model.eval()
    rows = []
    for batch_index, (images, labels, image_ids) in enumerate(loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        probabilities = torch.softmax(model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        batch_correct = predictions.eq(labels)

        for index in range(labels.size(0)):
            rows.append({
                "dataset": dataset_name,
                "model": model_name,
                "seed": seed,
                "image_id": int(image_ids[index].item()),
                "true_label": int(labels[index].item()),
                "predicted_label": int(predictions[index].item()),
                "correct": int(batch_correct[index].item()),
                "confidence": float(confidences[index].item()),
                "degradation": degradation,
                "severity": severity
            })
    if not rows:
        raise ValueError("Evaluation produced no predictions.")
    return rows