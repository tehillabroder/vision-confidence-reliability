"""Tests for the GTSRB validation profile."""

import pytest
import torch
import torch.nn as nn
from scripts.build_gtsrb_validation_profile import build_gtsrb_validation_profile, load_validation_model, validate_checkpoint_metadata


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

def test_load_validation_model_builds_configured_architecture(monkeypatch, tmp_path):
    # confirm validation rebuilds the configured model without downloading pretrained weights
    captured = {}
    model = nn.Linear(1, 1)

    def fake_model_builder(model_name, num_classes, pretrained_weights):
        captured["model_name"] = model_name
        captured["num_classes"] = num_classes
        captured["pretrained_weights"] = pretrained_weights
        return model

    def fake_checkpoint_loader(loaded_model, checkpoint_path, device):
        assert loaded_model is model
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        return {"model": "ResNet18"}

    monkeypatch.setattr("scripts.build_gtsrb_validation_profile.build_gtsrb_model", fake_model_builder)
    monkeypatch.setattr("scripts.build_gtsrb_validation_profile.load_model_checkpoint", fake_checkpoint_loader)

    checkpoint_path = tmp_path / "gtsrb_resnet18.pt"
    loaded_model, metadata = load_validation_model(checkpoint_path, torch.device("cpu"), "ResNet18")

    assert loaded_model is model
    assert metadata == {"model": "ResNet18"}
    assert captured["model_name"] == "ResNet18"
    assert captured["num_classes"] == 43
    assert captured["pretrained_weights"] is None
    assert captured["checkpoint_path"] == checkpoint_path
    assert captured["device"] == torch.device("cpu")
    assert not loaded_model.training

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

def test_validate_checkpoint_metadata_accepts_resnet18():
    # confirm the same checkpoint checks work for the selected stronger model
    config = valid_config()
    config["model"] = "ResNet18"
    metadata = valid_metadata()
    metadata["model"] = "ResNet18"

    validate_checkpoint_metadata(metadata, config, valid_split_metadata())

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