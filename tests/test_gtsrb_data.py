"""Tests for GTSRB dataset helpers."""

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.datasets.gtsrb import (
    GTSRB_CLASS_COUNT,
    GTSRB_NORMALISE,
    build_gtsrb_test_dataset,
    build_gtsrb_train_validation_datasets
)

class FakeGTSRB(Dataset):
    """
    Provide eight variable-sized traffic-sign images.
    Include class 42 to check the highest valid GTSRB label.
    """
    requests: list[tuple[str, bool]] = []

    def __init__(self, root: str, split: str, download: bool):
        self.root = root
        self.split = split
        self.requests.append((split, download))
        self.labels = [0, 1, 42, 3, 4, 5, 6, 7]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        # variable sizes represent the original GTSRB image format
        image = Image.new("RGB", (20 + index, 24 + index), (index * 20, 50, 100))
        return image, self.labels[index]

@pytest.fixture
def fake_gtsrb(monkeypatch):
    # replace the real dataset so tests do not depend on downloads
    FakeGTSRB.requests = []
    monkeypatch.setattr("src.datasets.gtsrb.datasets.GTSRB", FakeGTSRB)
    return FakeGTSRB

def test_gtsrb_train_validation_split_is_repeatable(tmp_path, fake_gtsrb):
    # check that the same seed produces the same fixed validation split
    first_train, first_validation = build_gtsrb_train_validation_datasets(
        str(tmp_path),
        validation_size=2,
        seed=42,
        download=False
    )
    second_train, second_validation = build_gtsrb_train_validation_datasets(
        str(tmp_path),
        validation_size=2,
        seed=42,
        download=False
    )

    assert first_train.indices == second_train.indices
    assert first_validation.indices == second_validation.indices
    assert len(first_train) == 6
    assert len(first_validation) == 2
    assert fake_gtsrb.requests == [("train", False), ("train", False)]

@pytest.mark.parametrize("validation_size", [-1, 0, 8])
def test_gtsrb_split_rejects_invalid_validation_size(tmp_path, fake_gtsrb, validation_size):
    # cover negative, empty and full-size validation splits
    with pytest.raises(ValueError, match="Validation size must be greater than zero"):
        build_gtsrb_train_validation_datasets(
            str(tmp_path),
            validation_size=validation_size,
            seed=42,
            download=False
        )

def test_gtsrb_test_dataset_returns_expected_sample(tmp_path, fake_gtsrb):
    # confirm evaluation data returns a normalised RGB tensor, label and stable image id
    dataset = build_gtsrb_test_dataset(str(tmp_path), download=False)
    image, label, image_id = dataset[2]

    assert GTSRB_CLASS_COUNT == 43
    # preprocessing should standardise variable source images to the model input shape
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    # normalisation may change the range but must not produce invalid values
    assert torch.isfinite(image).all()
    assert label == 42
    assert image_id == 2
    assert fake_gtsrb.requests == [("test", False)]

def test_gtsrb_degradation_occurs_before_normalisation(tmp_path, fake_gtsrb, monkeypatch):
    # confirm degradation receives resized image values between zero and one
    captured = {}

    def capture_degradation(image, degradation, severity):
        captured["image"] = image.clone()
        captured["degradation"] = degradation
        captured["severity"] = severity
        return torch.full_like(image, 0.25)

    monkeypatch.setattr("src.datasets.gtsrb.apply_degradation", capture_degradation)
    dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="low_light",
        severity=3,
        download=False
    )
    image, _, _ = dataset[0]
    expected = GTSRB_NORMALISE(torch.full((3, 64, 64), 0.25))

    assert captured["image"].shape == (3, 64, 64)
    assert captured["image"].min() >= 0
    assert captured["image"].max() <= 1
    assert captured["degradation"] == "low_light"
    assert captured["severity"] == 3
    # the returned tensor should only be normalised after degradation
    assert torch.allclose(image, expected)

def test_gtsrb_dataset_can_return_unnormalised_images(tmp_path, fake_gtsrb):
    # check that unnormalised images remain suitable for visual inspection
    dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="noise",
        severity=5,
        download=False,
        normalise=False
    )
    # fix the noise so this test remains repeatable
    torch.manual_seed(42)
    image, label, image_id = dataset[1]

    assert image.shape == (3, 64, 64)
    assert image.min() >= 0
    assert image.max() <= 1
    assert label == 1
    assert image_id == 1

def test_gtsrb_low_light_strengthens_with_severity(tmp_path, fake_gtsrb):
    # check that the severity scale produces progressively darker images
    mild_dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="low_light",
        severity=1,
        download=False,
        normalise=False
    )
    severe_dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="low_light",
        severity=5,
        download=False,
        normalise=False
    )
    # first _ ignores label, second _ ignores image identifier
    mild_image, _, _ = mild_dataset[4]
    severe_image, _, _ = severe_dataset[4]

    assert severe_image.mean() < mild_image.mean()