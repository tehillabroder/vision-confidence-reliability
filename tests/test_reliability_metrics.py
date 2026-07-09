"""Unit tests for calibration and reliability metrics."""

import pytest
from src.metrics.reliability import calibration_bins, expected_calibration_error, high_confidence_error_rate

def test_ece_is_zero_for_perfect_confident_predictions():
    # a perfectly confident and correct model should have zero calibration error
    correct = [1, 1, 1]
    confidences = [1.0, 1.0, 1.0]
    assert expected_calibration_error(correct, confidences) == pytest.approx(0.0)

def test_high_confidence_error_rate_counts_confident_errors():
    # one out of four predictions is a high confidence error (0.95 >= 0.90 threshold)
    correct = [1, 0, 0, 1]
    confidences = [0.95, 0.92, 0.40, 0.80]
    assert high_confidence_error_rate(correct, confidences, threshold=0.90) == 0.25

def test_calibration_bins_returns_requested_number_of_bins():
    correct = [1, 0, 1]
    confidences = [0.2, 0.6, 0.9]
    rows = calibration_bins(correct, confidences, n_bins=10)
    assert len(rows) == 10
    assert sum(row["count"] for row in rows) == 3

def test_ece_rejects_mismatched_lengths():
    # ensure metrics fail safely if prediction arrays are out of sync
    with pytest.raises(ValueError):
        expected_calibration_error([1, 0], [0.9])