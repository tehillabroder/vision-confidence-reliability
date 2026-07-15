"""Tests for model checkpoint handling."""

import pytest
import torch

from src.models.checkpoints import load_model_checkpoint, save_model_checkpoint
from src.models.simple_cnn import SimpleCNN

def test_checkpoint_round_trip_preserves_model_weights(tmp_path):
    # check that saved weights load back unchanged
    source_model = SimpleCNN()
    with torch.no_grad():
        # give the model known weights before saving
        next(source_model.parameters()).fill_(0.25)

    checkpoint_path = tmp_path / "model.pt"
    save_model_checkpoint(source_model, checkpoint_path, {"seed": 42})

    loaded_model = SimpleCNN()
    metadata = load_model_checkpoint(
        loaded_model,
        checkpoint_path,
        torch.device("cpu")
    )

    # zip pairs matching model weights together so the test can compare them one by one
    for source_parameter, loaded_parameter in zip(source_model.parameters(), loaded_model.parameters()):
        assert torch.equal(source_parameter, loaded_parameter)
    assert metadata["seed"] == 42

def test_checkpoint_loader_rejects_missing_file(tmp_path):
    # check that a missing checkpoint raises an error
    model = SimpleCNN()

    with pytest.raises(FileNotFoundError):
        load_model_checkpoint(
            model,
            tmp_path / "missing.pt",
            torch.device("cpu")
        )