"""Tests for MNIST dataset helpers."""

import pytest
import torch
from torch.utils.data import TensorDataset

from src.datasets.mnist import split_dataset

def test_split_dataset_is_repeatable():
    # check that the same seed gives the same split
    dataset = TensorDataset(torch.arange(20))
    first_train, first_validation = split_dataset(dataset, validation_size=5, seed=42)
    second_train, second_validation = split_dataset(dataset, validation_size=5, seed=42)

    assert first_train.indices == second_train.indices
    assert first_validation.indices == second_validation.indices

def test_split_dataset_rejects_invalid_validation_size():
    # check that an invalid validation size raises an error
    dataset = TensorDataset(torch.arange(20))

    with pytest.raises(ValueError):
        split_dataset(dataset, validation_size=20, seed=42)

def test_split_dataset_returns_expected_sizes():
    # check that the split sizes are correct
    dataset = TensorDataset(torch.arange(20))
    train_set, validation_set = split_dataset(dataset, validation_size=5, seed=42)

    assert len(train_set) == 15
    assert len(validation_set) == 5
