"""Tests for the validation reference profile."""

import json

import pytest
import torch
import torch.nn as nn

from src.evaluation.validation_profile import (
    build_validation_profile,
    calculate_adaptive_threshold,
    collect_validation_predictions,
    save_validation_profile,
    load_validation_profile,
    validate_validation_profile_source
)

class CountingModel(nn.Module):
    """Small model used to count validation calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        logits = torch.zeros((images.size(0), 10), device=images.device)
        # force a dominant logit so the softmax layer outputs completely stable confidences
        logits[:, 0] = 2.0
        return logits
    
def valid_profile() -> dict:
    return {
        "dataset": "MNIST",
        "model": "SimpleCNN",
        "checkpoint": "checkpoints/mnist_simple_cnn.pt",
        "seed": 42,
        "degradation": "none",
        "severity": 0,
        "validation_sample_count": 5000,
        "baseline_accuracy": 0.98,
        "baseline_mean_confidence": 0.97,
        "baseline_ece": 0.02,
        "baseline_confidence_accuracy_gap": -0.01,
        "baseline_fixed_hcer": 0.01,
        "fixed_hcer_threshold": 0.90,
        "adaptive_hcer_percentile": 90,
        "adaptive_hcer_threshold": 0.99
    }

def test_calculate_adaptive_threshold_uses_requested_percentile():
    # confirm the threshold comes from the complete confidence distribution
    confidences = [0.20, 0.70, 0.90, 0.95]

    threshold = calculate_adaptive_threshold(confidences, percentile=50)

    assert threshold == pytest.approx(0.80)

def test_calculate_adaptive_threshold_rejects_empty_confidences():
    # ensure an empty validation result cannot create a threshold
    with pytest.raises(ValueError, match="empty confidences"):
        calculate_adaptive_threshold([], percentile=90)

def test_collect_validation_predictions_runs_model_once_per_batch():
    # confirm validation uses one inference call for each batch
    model = CountingModel()
    loader = [(
        torch.zeros((2, 1, 28, 28)),
        torch.tensor([0, 1])
    )]

    correct, confidences = collect_validation_predictions(
        model,
        loader,
        torch.device("cpu")
    )

    assert model.calls == 1
    assert correct == [1, 0]
    assert len(confidences) == 2

def test_collect_validation_predictions_rejects_empty_loader():
    # ensure validation cannot silently produce an empty profile
    with pytest.raises(ValueError, match="no predictions"):
        collect_validation_predictions(
            CountingModel(),
            [],
            torch.device("cpu")
        )

def test_build_validation_profile_saves_baseline_metrics():
    # confirm the profile records its source and reliability settings
    correct = [1, 0, 1, 0]
    confidences = [0.95, 0.90, 0.70, 0.20]

    profile = build_validation_profile(
        dataset="MNIST",
        model="SimpleCNN",
        checkpoint="checkpoints/mnist_simple_cnn.pt",
        seed=42,
        correct=correct,
        confidences=confidences,
        ece_bins=10,
        fixed_hcer_threshold=0.90,
        adaptive_hcer_percentile=50
    )

    assert profile["degradation"] == "none"
    assert profile["validation_sample_count"] == 4
    assert profile["baseline_accuracy"] == 0.5
    assert profile["baseline_mean_confidence"] == pytest.approx(0.6875)
    assert profile["baseline_confidence_accuracy_gap"] == pytest.approx(0.1875)
    assert profile["baseline_fixed_hcer"] == 0.25
    assert profile["adaptive_hcer_threshold"] == pytest.approx(0.80)

def test_save_validation_profile_writes_json(tmp_path):
    # check that the reference profile is saved as reusable evidence
    profile = {"dataset": "MNIST", "adaptive_hcer_threshold": 0.95}
    output_path = tmp_path / "validation_profile.json"

    saved_path = save_validation_profile(profile, output_path)

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == profile

def test_load_validation_profile_returns_saved_profile(tmp_path):
    # confirm the saved threshold can be reused by evaluation
    profile = valid_profile()
    profile_path = tmp_path / "validation_profile.json"
    save_validation_profile(profile, profile_path)

    loaded_profile = load_validation_profile(profile_path)

    assert loaded_profile == profile

def test_load_validation_profile_rejects_invalid_adaptive_threshold(tmp_path):
    # ensure the adaptive threshold remains a valid probability
    profile = valid_profile()
    profile["adaptive_hcer_threshold"] = 1.1
    profile_path = tmp_path / "validation_profile.json"
    save_validation_profile(profile, profile_path)

    with pytest.raises(ValueError, match="adaptive_hcer_threshold"):
        load_validation_profile(profile_path)

def test_validate_validation_profile_source_rejects_mismatched_seed():
    # ensure a profile from another experiment cannot be reused silently
    profile = valid_profile()

    with pytest.raises(ValueError, match="seed"):
        validate_validation_profile_source(
            profile=profile,
            dataset="MNIST",
            model="SimpleCNN",
            checkpoint="checkpoints/mnist_simple_cnn.pt",
            seed=7,
            fixed_hcer_threshold=0.90,
            adaptive_hcer_percentile=90
        )