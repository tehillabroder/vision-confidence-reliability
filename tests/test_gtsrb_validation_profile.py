"""Tests for the GTSRB validation profile."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.build_gtsrb_validation_profile import (
    build_gtsrb_validation_profile,
    collect_gtsrb_validation_outputs,
    validate_checkpoint_metadata
)

class StaticModel(nn.Module):
    """Return fixed class predictions."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        logits = torch.full((len(predictions), 43), -5.0)
        logits[torch.arange(len(predictions)), predictions] = 5.0
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits[:images.size(0)]

def valid_config() -> dict:
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "seed": 42,
        "training": {
            "validation_size": 4000
        }
    }

def valid_metadata() -> dict:
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "seed": 42,
        "validation_size": 4000,
        "class_count": 43
    }

def test_collect_gtsrb_validation_outputs_returns_expected_values():
    # confirm labels, predictions and correctness are retained
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 2, 2, 1])
    image_ids = torch.arange(4)
    loader = DataLoader(TensorDataset(images, labels, image_ids), batch_size=4)
    model = StaticModel(torch.tensor([0, 1, 2, 3]))

    correct, true_labels, predicted_labels, confidences = collect_gtsrb_validation_outputs(
        model,
        loader,
        torch.device("cpu")
    )

    assert correct == [1, 0, 1, 0]
    assert true_labels == [0, 2, 2, 1]
    assert predicted_labels == [0, 1, 2, 3]
    assert len(confidences) == 4
    assert all(0 <= confidence <= 1 for confidence in confidences)

def test_collect_gtsrb_validation_outputs_rejects_empty_loader():
    # ensure empty validation data raises a clear error
    images = torch.empty(0, 3, 8, 8)
    labels = torch.empty(0, dtype=torch.long)
    image_ids = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, image_ids), batch_size=2)
    model = StaticModel(torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match="Validation loader produced no examples"):
        collect_gtsrb_validation_outputs(
            model,
            loader,
            torch.device("cpu")
        )

def test_build_gtsrb_validation_profile_adds_balanced_accuracy():
    # confirm the profile includes the GTSRB class-sensitive metric
    profile = build_gtsrb_validation_profile(
        dataset="GTSRB",
        model_name="GTSRBCNN",
        checkpoint="checkpoints/gtsrb_cnn.pt",
        seed=42,
        correct=[1, 1, 0, 0],
        true_labels=[0, 0, 0, 1],
        predicted_labels=[0, 0, 1, 0],
        confidences=[0.9, 0.8, 0.7, 0.6],
        ece_bins=10,
        fixed_hcer_threshold=0.90,
        adaptive_hcer_percentile=50
    )

    assert profile["dataset"] == "GTSRB"
    assert profile["degradation"] == "none"
    assert profile["severity"] == 0
    assert profile["validation_sample_count"] == 4
    assert profile["baseline_accuracy"] == pytest.approx(0.5)
    assert profile["baseline_balanced_accuracy"] == pytest.approx(1 / 3)
    assert profile["adaptive_hcer_threshold"] == pytest.approx(0.75)

def test_validate_checkpoint_metadata_accepts_matching_values():
    # confirm matching training evidence is accepted
    validate_checkpoint_metadata(valid_metadata(), valid_config())

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dataset", "MNIST"),
        ("model", "SimpleCNN"),
        ("seed", 7),
        ("validation_size", 100),
        ("class_count", 10)
    ]
)
def test_validate_checkpoint_metadata_rejects_mismatch(name, value):
    # ensure the profile cannot use a different training source
    metadata = valid_metadata()
    metadata[name] = value

    with pytest.raises(ValueError, match=f"Checkpoint metadata {name}"):
        validate_checkpoint_metadata(metadata, valid_config())