"""Tests for GTSRB training helpers."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.train_gtsrb import calculate_validation_metrics, train_model
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