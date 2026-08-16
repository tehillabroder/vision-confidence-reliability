"""Tests for the GTSRB validation profile."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.build_gtsrb_validation_profile import build_gtsrb_validation_profile, collect_gtsrb_validation_outputs, validate_checkpoint_metadata

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
            "validation_size": 4000,
            "validation_split": "stratified_track"
        }
    }

def valid_split_metadata() -> dict:
    return {
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "validation_size": 3990,
        "validation_size_difference": -10,
        "train_size": 22650,
        "track_size": 30,
        "total_track_count": 888,
        "train_track_count": 755,
        "validation_track_count": 133,
        "track_overlap": 0,
        "training_class_count": 43,
        "validation_class_count": 43,
        "validation_track_hash": "a" * 64
    }

def small_split_metadata() -> dict:
    return {
        "validation_split": "stratified_track",
        "requested_validation_size": 4,
        "validation_size": 4,
        "validation_size_difference": 0,
        "train_size": 4,
        "track_size": 1,
        "total_track_count": 8,
        "train_track_count": 4,
        "validation_track_count": 4,
        "track_overlap": 0,
        "training_class_count": 2,
        "validation_class_count": 2,
        "validation_track_hash": "b" * 64
    }

def valid_metadata() -> dict:
    split_metadata = valid_split_metadata()
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "seed": 42,
        "class_count": 43,
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "train_size": 22650,
        "validation_size": 3990,
        "track_overlap": 0,
        "validation_track_hash": "a" * 64,
        "split_metadata": split_metadata
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

def test_build_gtsrb_validation_profile_adds_split_evidence():
    # confirm the profile records class balance and track evidence
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
        adaptive_hcer_percentile=50,
        rank_hcer_top_fraction=0.50,
        split_metadata=small_split_metadata()
    )

    assert profile["validation_sample_count"] == 4
    assert profile["baseline_accuracy"] == pytest.approx(0.5)
    assert profile["baseline_balanced_accuracy"] == pytest.approx(1 / 3)
    assert profile["validation_split"] == "stratified_track"
    assert profile["track_overlap"] == 0
    assert profile["validation_track_hash"] == "b" * 64
    assert profile["baseline_adaptive_hcer"] == pytest.approx(0.0)
    assert profile["baseline_adaptive_hcer_coverage"] == pytest.approx(0.5)
    assert profile["baseline_rank_hcer"] == pytest.approx(0.0)
    assert profile["baseline_rank_hcer_coverage"] == pytest.approx(0.5)
    assert profile["rank_hcer_top_fraction"] == pytest.approx(0.5)

def test_validate_checkpoint_metadata_accepts_matching_split():
    # confirm matching track evidence is accepted
    validate_checkpoint_metadata(
        valid_metadata(),
        valid_config(),
        valid_split_metadata()
    )

def test_validate_checkpoint_metadata_rejects_split_hash_mismatch():
    # ensure a profile cannot use a checkpoint from different tracks
    metadata = valid_metadata()
    metadata["split_metadata"] = dict(
        metadata["split_metadata"]
    )
    metadata["split_metadata"]["validation_track_hash"] = "c" * 64

    with pytest.raises(
        ValueError,
        match="Checkpoint split metadata does not match"
    ):
        validate_checkpoint_metadata(
            metadata,
            valid_config(),
            valid_split_metadata()
        )