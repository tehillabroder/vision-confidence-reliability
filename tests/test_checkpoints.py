"""Tests for model checkpoint handling."""

import pytest
import torch

from src.models.checkpoints import load_model_checkpoint, save_model_checkpoint, validate_checkpoint_source
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

def test_checkpoint_loader_rejects_missing_model_state(tmp_path):
    # ensure a file without model weights cannot be treated as a checkpoint
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"metadata": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="model_state_dict"):
        load_model_checkpoint(SimpleCNN(), checkpoint_path, torch.device("cpu"))

def test_checkpoint_loader_rejects_invalid_model_state(tmp_path):
    # ensure incorrectly formed model weights fail with a clear checkpoint error
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": [], "metadata": {}}, checkpoint_path)

    with pytest.raises(ValueError, match="must be a mapping"):
        load_model_checkpoint(SimpleCNN(), checkpoint_path, torch.device("cpu"))

def test_checkpoint_loader_rejects_invalid_metadata(tmp_path):
    # makes sure that checkpoint metadata remains structured evidence
    checkpoint_path = tmp_path / "invalid.pt"
    torch.save({
        "model_state_dict": SimpleCNN().state_dict(),
        "metadata": []
    }, checkpoint_path)

    with pytest.raises(ValueError, match="metadata must be a dictionary"):
        load_model_checkpoint(SimpleCNN(), checkpoint_path, torch.device("cpu"))

def test_validate_checkpoint_source_accepts_matching_metadata():
    # confirm checkpoint identity can be checked against the configured run
    validate_checkpoint_source(
        {"dataset": "MNIST", "model": "SimpleCNN", "seed": 42},
        {"dataset": "MNIST", "model": "SimpleCNN", "seed": 42},
        "MNIST configuration"
    )

def test_validate_checkpoint_source_rejects_mismatched_metadata():
    # ensure a compatible model cannot be used with the wrong saved source
    with pytest.raises(ValueError, match="seed"):
        validate_checkpoint_source(
            {"dataset": "MNIST", "model": "SimpleCNN", "seed": 7},
            {"dataset": "MNIST", "model": "SimpleCNN", "seed": 42},
            "MNIST configuration"
        )

def test_validate_checkpoint_source_rejects_missing_metadata():
    # ensure incomplete checkpoint evidence cannot pass the source check
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_source(
            {"dataset": "MNIST", "seed": 42},
            {"dataset": "MNIST", "model": "SimpleCNN", "seed": 42},
            "MNIST configuration"
        )