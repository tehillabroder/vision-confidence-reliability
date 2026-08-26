"""Tests for MNIST training helpers."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scripts.train_mnist import build_checkpoint_metadata, train_model
from src.datasets.mnist import MNIST_TRAINING_AUGMENTATION

def test_train_model_uses_configured_learning_rate():
    # confirm changing the configured learning rate changes the training update
    torch.manual_seed(42)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.tensor([0, 1, 2, 3])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    source_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    slow_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    fast_model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
    slow_model.load_state_dict(source_model.state_dict())
    fast_model.load_state_dict(source_model.state_dict())

    # independent frozen copy of the layer's weights for checking against after training
    initial_weights = source_model[1].weight.detach().clone()

    train_model(slow_model, loader, torch.device("cpu"), epochs=1, learning_rate=0.001, max_train_batches=1)
    train_model(fast_model, loader, torch.device("cpu"), epochs=1, learning_rate=0.01, max_train_batches=1)

    slow_change = torch.linalg.vector_norm(slow_model[1].weight.detach() - initial_weights)
    fast_change = torch.linalg.vector_norm(fast_model[1].weight.detach() - initial_weights)

    assert slow_change > 0
    assert fast_change > slow_change

def test_build_checkpoint_metadata_records_training_context():
    # confirm future MNIST checkpoints record the settings that produced them
    config = {
        "dataset": "MNIST",
        "model": "SimpleCNN",
        "seed": 42,
        "training": {
            "epochs": 1,
            "batch_size": 64,
            "learning_rate": 0.001
        }
    }

    metadata = build_checkpoint_metadata(
        config=config,
        train_size=55000,
        validation_size=5000,
        validation_accuracy=0.98,
        device=torch.device("cpu"),
        config_copy_path=Path("checkpoints/mnist_simple_cnn_config.yaml")
    )

    assert metadata["learning_rate"] == pytest.approx(0.001)
    assert metadata["image_size"] == (28, 28)
    assert metadata["normalisation_mean"] == [0.1307]
    assert metadata["normalisation_std"] == [0.3081]
    assert metadata["training_augmentation"] == MNIST_TRAINING_AUGMENTATION
    assert metadata["device"] == "cpu"