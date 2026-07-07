"""Unit tests for basic reliability metric functions."""

import pytest

from src.metrics.basic import (
    accuracy_from_correct,
    confidence_accuracy_gap,
    mean_confidence,
)

def test_accuracy_from_correct():
    # confirm 3 out of 4 correct scores returns 75% accuracy
    correct = [1, 1, 0, 1]
    assert accuracy_from_correct(correct) == 0.75

def test_mean_confidence():
    # check average calculation handles repeating decimals correctly
    confidences = [0.8, 0.6, 0.9]
    assert mean_confidence(confidences) == pytest.approx(0.766666, abs=1e-5)

def test_confidence_accuracy_gap():
    # check that 80% mean confidence minus 50% accuracy equals a 0.3 gap
    correct = [1, 0]
    confidences = [0.9, 0.7]
    assert confidence_accuracy_gap(correct, confidences) == pytest.approx(0.3)

def test_confidence_values_must_be_valid():
    # ensure values outside the 0 to 1 probability range trigger an error
    with pytest.raises(ValueError):
        mean_confidence([0.5, 1.2])