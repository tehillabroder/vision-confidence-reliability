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
    # confirm stable sorting picks exactly the top 40% even when scores tie
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

def test_ece_handles_empty_bins_gracefully():
    # polarised predictions should leave the middle bins completely empty
    correct = [1, 0]
    confidences = [0.1, 0.9]
    # this proves the loop continues safely without triggering divide-by-zero errors
    ece = expected_calibration_error(correct, confidences, n_bins=10)
    assert ece == pytest.approx(0.9)

@pytest.mark.parametrize("metric", [expected_calibration_error, high_confidence_error_rate, calibration_bins])
def test_prediction_metrics_reject_mismatched_lengths(metric):
    # reliability metrics must not use prediction arrays that are out of sync
    with pytest.raises(ValueError, match="same length"):
        metric([1, 0], [0.9])

@pytest.mark.parametrize("metric", [expected_calibration_error, high_confidence_error_rate, calibration_bins])
def test_prediction_metrics_reject_empty_inputs(metric):
    # checks that condition metrics cannot be calculated without predictions
    with pytest.raises(ValueError):
        metric([], [])

@pytest.mark.parametrize("metric", [expected_calibration_error, high_confidence_error_rate, calibration_bins])
def test_prediction_metrics_reject_invalid_confidences(metric):
    # check confidence values remain valid probabilities
    with pytest.raises(ValueError, match="between 0 and 1"):
        metric([1], [1.5])

def test_hcer_rejects_invalid_threshold():
    # ensure HCER cannot use an invalid confidence threshold
    with pytest.raises(ValueError, match="between 0 and 1"):
        high_confidence_error_rate([1], [0.9], threshold=1.1)

def test_calibration_bins_rejects_invalid_bin_count():
    # calibration output must always use a real binning scheme
    with pytest.raises(ValueError, match="positive integer"):
        calibration_bins([1], [0.9], n_bins=0)

def test_rank_based_hcer_rejects_invalid_correctness():
    # ensure ranked HCER still requires binary correctness values
    with pytest.raises(ValueError, match="0 or 1"):
        rank_based_high_confidence_error_rate([2, 0], [0.9, 0.8])