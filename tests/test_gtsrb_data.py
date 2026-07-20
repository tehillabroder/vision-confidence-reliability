"""Tests for GTSRB dataset helpers."""

import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.gtsrb import GTSRB_CLASS_COUNT, build_gtsrb_test_dataset, build_gtsrb_train_validation_datasets

class FakeGTSRB(Dataset):
    """
    Create a tiny dummy dataset of 8 fake images.
    Simulate GTSRB's variable image dimensions (20+index x 24+index).
    Assign realistic traffic sign class labels (incl class 42, the maximum class index out of 43).
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
    FakeGTSRB.requests = []
    monkeypatch.setattr("src.datasets.gtsrb.datasets.GTSRB", FakeGTSRB)
    return FakeGTSRB

def test_gtsrb_train_validation_split_is_repeatable(tmp_path, fake_gtsrb):
    # check that the same seed gives the same split
    first_train, first_validation = build_gtsrb_train_validation_datasets(
        str(tmp_path),
        validation_size=2,
        seed=42,
        # confirm no real network downloads were attempted
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
    # test three invalid validation set sizes: 
    # -1 (negative), 0 (empty), and 8 (equal to the total dataset size, leaving 0 for training).
    # check that invalid validation sizes raise a clear error
    with pytest.raises(ValueError, match="Validation size must be greater than zero"):
        build_gtsrb_train_validation_datasets(
            str(tmp_path),
            validation_size=validation_size,
            seed=42,
            download=False
        )

def test_gtsrb_test_dataset_returns_expected_sample(tmp_path, fake_gtsrb):
    # confirm that evaluation data returns a normalised RGB tensor, label and image id
    dataset = build_gtsrb_test_dataset(str(tmp_path), download=False)
    image, label, image_id = dataset[2]

    assert GTSRB_CLASS_COUNT == 43
    # image should be a PyTorch float32 tensor with 3 RGB channels, 64x64 pixels
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    # all pixel values should be valid (no NaN or Inf values after normalisation)
    assert torch.isfinite(image).all()
    assert label == 42
    assert image_id == 2
    assert fake_gtsrb.requests == [("test", False)]