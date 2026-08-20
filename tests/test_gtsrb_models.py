"""Tests for configurable GTSRB models."""

import pytest
import torch
import torch.nn as nn
from src.models.checkpoints import load_model_checkpoint, save_model_checkpoint
from src.models.gtsrb_models import build_gtsrb_model

@pytest.mark.parametrize("model_name", ["ResNet18", "MobileNetV2"])
def test_gtsrb_torchvision_models_return_one_score_per_class(model_name):
    # confirm each stronger model returns scores for all 43 classes
    # use random weights so unit tests run fast without downloading large files
    model = build_gtsrb_model(model_name, num_classes=43, pretrained_weights=None)
    model.eval()
    outputs = model(torch.randn(2, 3, 64, 64))

    assert outputs.shape == (2, 43)

@pytest.mark.parametrize("model_name", ["ResNet18", "MobileNetV2"])
def test_gtsrb_torchvision_models_support_backpropagation(model_name):
    # nn.Linear(in_features, 43) replaced the final classification head with a custom linear layer outputting 43 classes
    # check that the stronger models remain trainable after head replacement
    model = build_gtsrb_model(model_name, num_classes=43, pretrained_weights=None)
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(model(images), labels)

    loss.backward()

    # makes sure that replacing the final layer didn't accidentally freeze the rest of the network
    assert all(parameter.requires_grad for parameter in model.parameters())
    # and that loss.backward() actually computed and stored gradients in the network's layers
    assert any(parameter.grad is not None for parameter in model.parameters())

@pytest.mark.parametrize("model_name", ["ResNet18", "MobileNetV2"])
def test_gtsrb_torchvision_model_checkpoint_round_trip(tmp_path, model_name):
    # confirm saved stronger models reload with valid softmax probabilities
    images = torch.randn(2, 3, 64, 64)  # evaluating two sample images at once
    model = build_gtsrb_model(model_name, num_classes=43, pretrained_weights=None)
    model.eval()
    checkpoint_path = tmp_path / f"{model_name}.pt"

    with torch.no_grad():
        expected = torch.softmax(model(images), dim=1)

    save_model_checkpoint(model, checkpoint_path, {"model": model_name})
    loaded_model = build_gtsrb_model(model_name, num_classes=43, pretrained_weights=None)
    metadata = load_model_checkpoint(loaded_model, checkpoint_path, torch.device("cpu"))
    loaded_model.eval()

    with torch.no_grad():
        actual = torch.softmax(loaded_model(images), dim=1)

    assert metadata["model"] == model_name
    # the reloaded model should be producing identical probabilities from the same inputs
    assert torch.allclose(expected, actual)
    assert torch.all(actual >= 0)
    assert torch.all(actual <= 1)
    # predicted probabilities over all classes must add up to exactly 1.0 for each of the two images in the batch
    assert torch.allclose(actual.sum(dim=1), torch.ones(2))

def test_gtsrb_model_builder_rejects_unknown_model():
    # ensure unsupported model names fail clearly
    with pytest.raises(ValueError, match="Unsupported GTSRB model"):
        build_gtsrb_model("UnknownModel")

def test_gtsrb_cnn_rejects_pretrained_weights():
    # ensure the custom baseline cannot silently receive unrelated weights
    with pytest.raises(ValueError, match="does not support pretrained weights"):
        build_gtsrb_model("GTSRBCNN", pretrained_weights="IMAGENET1K_V1")