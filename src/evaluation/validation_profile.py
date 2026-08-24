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
def collect_validation_predictions(model: nn.Module, loader: Iterable, device: torch.device) -> tuple[list[int], list[int], list[int], list[float]]:
    model.eval()
    correct = []
    true_labels = []
    predicted_labels = []
    confidences = []

    for batch in loader:
        # image IDs are allowed so datasets can use the same validation collector
        if not isinstance(batch, (tuple, list)) or len(batch) not in (2, 3):
            raise ValueError("Validation batches must contain images and labels, with an optional image ID.")

        images, labels = batch[:2]
        images = images.to(device)
        labels = labels.to(device)
        probabilities = F.softmax(model(images), dim=1)
        batch_confidences, predictions = probabilities.max(dim=1)

        # keep the labels so dataset-specific metrics can be calculated later
        correct.extend(predictions.eq(labels).int().cpu().tolist())
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        confidences.extend(batch_confidences.cpu().tolist())

    if not correct:
        raise ValueError("Validation evaluation produced no predictions.")

    return correct, true_labels, predicted_labels, confidences

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

def load_validation_profile(profile_path: Path) -> dict:
    if not profile_path.exists():
        raise FileNotFoundError(f"Validation profile not found: {profile_path}")

    try:
        with profile_path.open("r", encoding="utf-8") as profile_file:
            profile = json.load(profile_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid validation profile JSON: {profile_path}") from error

    if not isinstance(profile, dict):
        raise ValueError("Validation profile must contain a dictionary.")

    required_fields = (
        "dataset",
        "model",
        "checkpoint",
        "seed",
        "degradation",
        "severity",
        "validation_sample_count",
        "baseline_accuracy",
        "baseline_mean_confidence",
        "baseline_ece",
        "baseline_confidence_accuracy_gap",
        "baseline_fixed_hcer",
        "fixed_hcer_threshold",
        "adaptive_hcer_percentile",
        "adaptive_hcer_threshold"
    )
    missing_fields = [field for field in required_fields if field not in profile]
    if missing_fields:
        raise ValueError(f"Validation profile is missing: {', '.join(missing_fields)}.")

    if profile["degradation"] != "none" or profile["severity"] != 0:
        raise ValueError("Validation profile must use the undegraded condition.")

    sample_count = profile["validation_sample_count"]
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("Validation sample count must be a positive integer.")

    for field in ("fixed_hcer_threshold", "adaptive_hcer_threshold"):
        value = profile[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between 0 and 1.")

    percentile = profile["adaptive_hcer_percentile"]
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)) or not 0.0 <= percentile <= 100.0:
        raise ValueError("adaptive_hcer_percentile must be between 0 and 100.")

    return profile

def validate_validation_profile_source(
    profile: dict,
    dataset: str,
    model: str,
    checkpoint: str,
    seed: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_percentile: float
) -> None:
    if profile["dataset"] != dataset:
        raise ValueError("Validation profile dataset does not match the experiment.")
    if profile["model"] != model:
        raise ValueError("Validation profile model does not match the experiment.")
    if Path(profile["checkpoint"]) != Path(checkpoint):
        raise ValueError("Validation profile checkpoint does not match the experiment.")
    if profile["seed"] != seed:
        raise ValueError("Validation profile seed does not match the experiment.")
    if profile["fixed_hcer_threshold"] != fixed_hcer_threshold:
        raise ValueError("Validation profile fixed HCER threshold does not match the configuration.")
    if profile["adaptive_hcer_percentile"] != adaptive_hcer_percentile:
        raise ValueError("Validation profile adaptive percentile does not match the configuration.")