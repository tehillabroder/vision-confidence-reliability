"""Shared helpers for degradation evaluation."""

from typing import Iterable, Optional
import torch
import torch.nn as nn

def build_experiment_conditions(degradations: list[str], severity_levels: list[int]) -> list[tuple[str, int]]:
    conditions = [("none", 0)]
    for degradation in degradations:
        for severity in severity_levels:
            conditions.append((degradation, severity))
    return conditions

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