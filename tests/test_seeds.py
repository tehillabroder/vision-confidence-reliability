"""Tests for repeatable random seeds."""

import random
import numpy as np
import torch

from src.utils.seeds import set_seed

def test_set_seed_repeats_random_values():
    # check that each random generator repeats its values
    set_seed(42)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    set_seed(42)
    second_python = random.random()
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert np.array_equal(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)