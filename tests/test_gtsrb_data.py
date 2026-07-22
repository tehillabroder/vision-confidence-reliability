"""Tests for GTSRB dataset helpers."""

from pathlib import Path

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
    """Provide small variable-sized traffic-sign tracks."""

    requests: list[tuple[str, bool]] = []

    def __init__(self, root: str, split: str, download: bool):
        self.root = root
        self.split = split
        self.requests.append((split, download))
        self._samples = []
        self.labels = []

        for label in (0, 1, 42):
            for track in range(2):
                for frame in range(2):
                    image_path = (
                        Path(root)
                        / f"{label:05d}"
                        / f"{track:05d}_{frame:05d}.ppm"
                    )
                    self._samples.append((str(image_path), label))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image = Image.new(
            "RGB",
            (20 + index, 24 + index),
            (index * 20, 50, 100)
        )
        return image, self.labels[index]

@pytest.fixture
def fake_gtsrb(monkeypatch):
    # replace the real dataset so tests do not depend on downloads
    FakeGTSRB.requests = []
    monkeypatch.setattr(
        "src.datasets.gtsrb.datasets.GTSRB",
        FakeGTSRB
    )
    return FakeGTSRB

def test_gtsrb_train_validation_split_is_repeatable(tmp_path, fake_gtsrb):
    # check that the same seed produces the same fixed track split
    first_train, first_validation = build_gtsrb_train_validation_datasets(
        str(tmp_path),
        validation_size=6,
        seed=42,
        download=False
    )
    second_train, second_validation = build_gtsrb_train_validation_datasets(
        str(tmp_path),
        validation_size=6,
        seed=42,
        download=False
    )

    assert first_train.indices == second_train.indices
    assert first_validation.indices == second_validation.indices
    assert len(first_train) == 6
    assert len(first_validation) == 6
    assert fake_gtsrb.requests == [
        ("train", False),
        ("train", False)
    ]

@pytest.mark.parametrize("validation_size", [-1, 0, 12])
def test_gtsrb_split_rejects_invalid_validation_size(
    tmp_path,
    fake_gtsrb,
    validation_size
):
    # cover negative, empty and full-size validation splits
    with pytest.raises(ValueError, match="Validation size must be greater"):
        build_gtsrb_train_validation_datasets(
            str(tmp_path),
            validation_size=validation_size,
            seed=42,
            download=False
        )

def test_gtsrb_test_dataset_returns_expected_sample(tmp_path, fake_gtsrb):
    # confirm evaluation data returns a normalised RGB tensor and identifiers
    dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        download=False
    )
    image, label, image_id = dataset[8]

    assert GTSRB_CLASS_COUNT == 43
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert label == 42
    assert image_id == 8
    assert fake_gtsrb.requests == [("test", False)]

def test_gtsrb_degradation_occurs_before_normalisation(
    tmp_path,
    fake_gtsrb,
    monkeypatch
):
    # confirm degradation receives resized values between zero and one
    captured = {}

    def capture_degradation(image, degradation, severity):
        captured["image"] = image.clone()
        captured["degradation"] = degradation
        captured["severity"] = severity
        return torch.full_like(image, 0.25)

    monkeypatch.setattr(
        "src.datasets.gtsrb.apply_degradation",
        capture_degradation
    )
    dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="low_light",
        severity=3,
        download=False
    )
    image, _, _ = dataset[0]
    expected = GTSRB_NORMALISE(
        torch.full((3, 64, 64), 0.25)
    )

    assert captured["image"].shape == (3, 64, 64)
    assert captured["image"].min() >= 0
    assert captured["image"].max() <= 1
    assert captured["degradation"] == "low_light"
    assert captured["severity"] == 3
    assert torch.allclose(image, expected)

def test_gtsrb_dataset_can_return_unnormalised_images(
    tmp_path,
    fake_gtsrb
):
    # check that unnormalised images remain suitable for inspection
    dataset = build_gtsrb_test_dataset(
        str(tmp_path),
        degradation="noise",
        severity=5,
        download=False,
        normalise=False
    )
    torch.manual_seed(42)
    image, label, image_id = dataset[4]

    assert image.shape == (3, 64, 64)
    assert image.min() >= 0
    assert image.max() <= 1
    assert label == 1
    assert image_id == 4

def test_gtsrb_low_light_strengthens_with_severity(
    tmp_path,
    fake_gtsrb
):
    # check that higher severity produces a darker image
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

    mild_image, _, _ = mild_dataset[4]
    severe_image, _, _ = severe_dataset[4]

    assert severe_image.mean() < mild_image.mean()