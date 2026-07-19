"""Build undegraded validation reference profiles."""

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.metrics.reliability import expected_calibration_error, high_confidence_error_rate

@torch.no_grad()
def collect_validation_predictions(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[list[int], list[float]]:
    model.eval()
    correct = []
    confidences = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        probabilities = F.softmax(model(images), dim=1)
        batch_confidences, predictions = probabilities.max(dim=1)

        # cast to integer mask on cpu to ensure reliable binary mapping downstream
        correct.extend(predictions.eq(labels).int().cpu().tolist())
        confidences.extend(batch_confidences.cpu().tolist())

    if not correct:
        raise ValueError("Validation evaluation produced no predictions.")

    return correct, confidences

def calculate_adaptive_threshold(confidences: list[float], percentile: float) -> float:
    if not confidences:
        raise ValueError("Cannot calculate an adaptive threshold from empty confidences.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("Adaptive HCER percentile must be between 0 and 100.")

    confidence_array = np.asarray(confidences, dtype=float)
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")

    # keep the percentile calculation rule fixed across experiments
    return float(np.percentile(confidence_array, percentile, method="linear"))

def build_validation_profile(
    dataset: str,
    model: str,
    checkpoint: str,
    seed: int,
    correct: list[int],
    confidences: list[float],
    ece_bins: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_percentile: float
) -> dict:
    if not correct:
        raise ValueError("Cannot build a validation profile from empty predictions.")
    if len(correct) != len(confidences):
        raise ValueError("Correct and confidence arrays must have the same length.")

    adaptive_threshold = calculate_adaptive_threshold(confidences, adaptive_hcer_percentile)

    return {
        "dataset": dataset,
        "model": model,
        "checkpoint": checkpoint,
        "seed": seed,
        "degradation": "none",
        "severity": 0,
        "validation_sample_count": len(correct),
        "baseline_accuracy": accuracy_from_correct(correct),
        "baseline_mean_confidence": mean_confidence(confidences),
        "baseline_ece": expected_calibration_error(correct, confidences, n_bins=ece_bins),
        "baseline_confidence_accuracy_gap": confidence_accuracy_gap(correct, confidences),
        "baseline_fixed_hcer": high_confidence_error_rate(
            correct,
            confidences,
            threshold=fixed_hcer_threshold
        ),
        "fixed_hcer_threshold": fixed_hcer_threshold,
        "adaptive_hcer_percentile": adaptive_hcer_percentile,
        "adaptive_hcer_threshold": adaptive_threshold
    }

def save_validation_profile(profile: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(profile, output_file, indent=2)

    return output_path