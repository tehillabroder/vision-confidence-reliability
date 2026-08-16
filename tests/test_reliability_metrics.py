"""Unit tests for calibration and reliability metrics."""

import pytest
from src.metrics.basic import confidence_accuracy_gap
from src.metrics.reliability import calibration_bins, expected_calibration_error, high_confidence_coverage, high_confidence_error_rate, rank_based_high_confidence_coverage, rank_based_high_confidence_error_rate

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

def test_high_confidence_coverage_counts_selected_predictions():
    # confirm coverage shows how many predictions actually meet the threshold
    confidences = [1.0, 0.99, 0.90, 0.80]
    assert high_confidence_coverage(confidences, threshold=1.0) == pytest.approx(0.25)

def test_rank_based_hcer_keeps_fixed_coverage_with_ties():
    # comfirm that stable sorting picks exactly the top 40% (2 of 5 items) even when scores tie    
    correct = [1, 0, 0, 0, 1]
    confidences = [1.0, 1.0, 1.0, 1.0, 0.50]

    coverage = rank_based_high_confidence_coverage(confidences, top_fraction=0.40)
    hcer = rank_based_high_confidence_error_rate(correct, confidences, top_fraction=0.40)

    assert coverage == pytest.approx(0.40)
    # one selected error divided by all five predictions gives 0.20
    assert hcer == pytest.approx(0.20)

def test_ece_retains_size_but_not_calibration_direction():
    # confirm equal ECE can represent opposite confidence errors
    overconfident_correct = [1, 0, 0, 0]
    underconfident_correct = [1, 1, 1, 0]
    overconfident = [0.75, 0.75, 0.75, 0.75]
    underconfident = [0.25, 0.25, 0.25, 0.25]

    assert expected_calibration_error(overconfident_correct, overconfident) == pytest.approx(0.50)
    assert expected_calibration_error(underconfident_correct, underconfident) == pytest.approx(0.50)
    assert confidence_accuracy_gap(overconfident_correct, overconfident) == pytest.approx(0.50)
    assert confidence_accuracy_gap(underconfident_correct, underconfident) == pytest.approx(-0.50)

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

def test_ece_raises_error_for_empty_lists():
    # ensure metrics fail safely if provided with no data
    with pytest.raises(ValueError):
        expected_calibration_error([], [])

def test_ece_raises_error_for_invalid_confidence_values():
    # confidence probabilities must mathematically be between 0 and 1
    with pytest.raises(ValueError):
        expected_calibration_error([1], [1.5])

def test_ece_handles_empty_bins_gracefully():
    # polarised predictions should leave the middle bins completely empty
    correct = [1, 0]
    confidences = [0.1, 0.9]
    # this proves the loop continues safely without triggering divide-by-zero errors
    ece = expected_calibration_error(correct, confidences, n_bins=10)
    assert ece == pytest.approx(0.9)