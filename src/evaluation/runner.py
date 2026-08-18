"""Shared helpers for degradation evaluation."""

from pathlib import Path
from typing import Iterable, Optional
import torch
import torch.nn as nn
import pandas as pd
from src.metrics.reliability import calibration_bins
from src.utils.config import save_config_copy

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

def save_core_evaluation_outputs(
    prediction_rows: list[dict],
    metric_rows: list[dict],
    calibration_rows: list[dict],
    config_path: Path,
    output_dir: Path
) -> dict[str, Path]:
    # no empty-output validation here 
    # GTSRB already has that check, while MNIST currently does not
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "predictions.csv",
        "metrics": output_dir / "metrics_summary.csv",
        "calibration": output_dir / "calibration_bins.csv",
        "config": output_dir / "config.yaml"
    }

    pd.DataFrame(prediction_rows).to_csv(paths["predictions"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(calibration_rows).to_csv(paths["calibration"], index=False)

    save_config_copy(config_path, paths["config"])

    return paths

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