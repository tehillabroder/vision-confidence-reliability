"""Unit tests for image degradation functions."""

import pytest
import torch
from src.degradations.image_degradations import apply_degradation

def test_clean_severity_zero_returns_same_image():
    # confirm clean severity 0 leaves the image unchanged
    image = torch.rand(1, 28, 28)  # one MNIST-sized greyscale image
    degraded = apply_degradation(image, "clean", 0)
    assert torch.equal(image, degraded)

def test_invalid_degradation_raises_error():
    # ensure an unknown degradation name triggers an error
    image = torch.rand(1, 28, 28)
    with pytest.raises(ValueError):
        apply_degradation(image, "bad_degradation", 1)

def test_invalid_severity_raises_error():
    # ensure severity above the 0 to 5 range triggers an error
    image = torch.rand(1, 28, 28)
    with pytest.raises(ValueError):
        apply_degradation(image, "blur", 6)

def test_low_light_reduces_brightness():
    # check that severity 5 is darker than severity 1
    image = torch.ones(1, 28, 28)
    mild = apply_degradation(image, "low_light", 1)
    severe = apply_degradation(image, "low_light", 5)
    assert severe.mean() < mild.mean()

def test_noise_keeps_values_between_zero_and_one():
    # check noisy images remain within the valid image range
    torch.manual_seed(42)  # fixed seed makes random noise repeatable
    image = torch.ones(1, 28, 28) * 0.5  # mid-grey MNIST-sized image
    noisy = apply_degradation(image, "noise", 5)
    assert noisy.min() >= 0
    assert noisy.max() <= 1