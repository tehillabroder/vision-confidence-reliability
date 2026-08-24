"""Tests for GTSRB training helpers."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.train_gtsrb import build_checkpoint_metadata, calculate_validation_metrics, train_model

class StaticModel(nn.Module):
    """Return fixed predictions for validation metric tests."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        logits = torch.zeros(len(predictions), 43)
        logits[torch.arange(len(predictions)), predictions] = 1.0
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits[:images.size(0)]

def test_calculate_validation_metrics_returns_known_values():
    # confirm accuracy and balanced accuracy match a known example
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 0, 0, 1])
    image_ids = torch.arange(4)
    loader = DataLoader(TensorDataset(images, labels, image_ids), batch_size=4)
    model = StaticModel(torch.tensor([0, 0, 1, 0]))

    accuracy, balanced_accuracy = calculate_validation_metrics(
        model,
        loader,
        torch.device("cpu")
    )

    assert accuracy == pytest.approx(0.5)
    assert balanced_accuracy == pytest.approx(1 / 3)

def test_train_model_updates_weights():
    # check that one training batch changes the model weights
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    image_ids = torch.arange(4)
    loader = DataLoader(TensorDataset(images, labels, image_ids), batch_size=4)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 43))
    original_weights = model[1].weight.detach().clone()

    train_model(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=0.001,
        max_train_batches=1
    )

    assert not torch.equal(original_weights, model[1].weight)

def test_calculate_validation_metrics_rejects_empty_loader():
    # ensure empty validation data raises a clear error
    images = torch.empty(0, 3, 8, 8)
    labels = torch.empty(0, dtype=torch.long)
    image_ids = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels, image_ids), batch_size=2)
    model = StaticModel(torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match="Validation loader produced no examples"):
        calculate_validation_metrics(model, loader, torch.device("cpu"))

def test_build_checkpoint_metadata_records_reproducible_training_context():
    # confirm checkpoint evidence records the model, preprocessing and track split
    config = {
        "dataset": "GTSRB",
        "model": "ResNet18",
        "seed": 42,
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "learning_rate": 0.001,
            "pretrained_weights": "IMAGENET1K_V1",
            "training_strategy": "full_finetune",
            "validation_size": 4000,
            "validation_split": "stratified_track",
            "augmentation": {
                "resize": True
            }
        }
    }
    split_metadata = {
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

    metadata = build_checkpoint_metadata(
        config=config,
        split_metadata=split_metadata,
        validation_accuracy=0.90,
        validation_balanced_accuracy=0.88,
        device=torch.device("cpu"),
        config_copy_path=Path("checkpoints/config.yaml")
    )

    assert metadata["requested_validation_size"] == 4000
    assert metadata["validation_size"] == 3990
    assert metadata["track_overlap"] == 0
    assert metadata["validation_track_hash"] == "a" * 64
    assert metadata["split_metadata"] == split_metadata
    assert metadata["model"] == "ResNet18"
    assert metadata["pretrained_weights"] == "IMAGENET1K_V1"
    assert metadata["training_strategy"] == "full_finetune"
    assert metadata["image_size"] == (64, 64)
    assert metadata["normalisation_mean"] == [0.485, 0.456, 0.406]
    assert metadata["normalisation_std"] == [0.229, 0.224, 0.225]
    assert metadata["preprocessing_order"] == "resize_degrade_normalise"
    assert metadata["resize_interpolation"] == "bilinear"
    assert metadata["resize_antialias"] is True