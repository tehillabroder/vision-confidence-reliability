"""Reliability metrics for confidence evaluation."""

import numpy as np

def _validate_confidences(confidences) -> np.ndarray:
    confidence_array = np.asarray(confidences, dtype=float)

    if confidence_array.size == 0:
        raise ValueError("Confidence inputs must not be empty.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")

    return confidence_array

def _validate_prediction_inputs(correct, confidences) -> tuple[np.ndarray, np.ndarray]:
    correct_array = np.asarray(correct)
    confidence_array = _validate_confidences(confidences)

    if correct_array.size != confidence_array.size:
        raise ValueError("Correct and confidence arrays must have the same length.")
    if not np.all(np.isin(correct_array, [0, 1])):
        raise ValueError("Correct values must be 0 or 1.")

    return correct_array.astype(float), confidence_array

def _validate_bin_count(n_bins: int) -> None:
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("Calibration bin count must be a positive integer.")

def _validate_threshold(threshold: float) -> None:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")


def expected_calibration_error(correct, confidences, n_bins: int = 10) -> float:
    correct_array, confidence_array = _validate_prediction_inputs(correct, confidences)
    _validate_bin_count(n_bins)

    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins

        # keep exact edge values in one bin only
        if bin_index == 0:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array > lower) & (confidence_array <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = correct_array[in_bin].mean()
        bin_confidence = confidence_array[in_bin].mean()
        bin_weight = in_bin.mean()
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)

def high_confidence_error_rate(correct, confidences, threshold: float = 0.90) -> float:
    correct_array, confidence_array = _validate_prediction_inputs(correct, confidences)
    _validate_threshold(threshold)

    high_confidence_wrong = (correct_array == 0) & (confidence_array >= threshold)
    # average across all samples so that HCER measures overall risk
    return float(high_confidence_wrong.mean())

def high_confidence_coverage(confidences, threshold: float = 0.90) -> float:
    confidence_array = _validate_confidences(confidences)
    _validate_threshold(threshold)
    return float((confidence_array >= threshold).mean())

def _rank_based_selection_mask(confidences, top_fraction: float) -> np.ndarray:
    confidence_array = _validate_confidences(confidences)

    if isinstance(top_fraction, bool) or not isinstance(top_fraction, (int, float)) or not 0 < top_fraction <= 1:
        raise ValueError("Top confidence fraction must be greater than 0 and at most 1.")

    selected_count = max(1, int(np.ceil(confidence_array.size * top_fraction)))

    # stable sorting makes sure that identical scores sort consistently
    ranked_indices = np.argsort(-confidence_array, kind="stable")
    selected = np.zeros(confidence_array.size, dtype=bool)
    selected[ranked_indices[:selected_count]] = True
    return selected

def rank_based_high_confidence_error_rate(correct, confidences, top_fraction: float = 0.10) -> float:
    correct_array, confidence_array = _validate_prediction_inputs(correct, confidences)
    selected = _rank_based_selection_mask(confidence_array, top_fraction)
    high_confidence_wrong = (correct_array == 0) & selected

    # keep all predictions as the denominator so this stays comparable with fixed HCER
    return float(high_confidence_wrong.mean())

def rank_based_high_confidence_coverage(confidences, top_fraction: float = 0.10) -> float:
    selected = _rank_based_selection_mask(confidences, top_fraction)
    return float(selected.mean())

def calibration_bins(correct, confidences, n_bins: int = 10) -> list[dict]:
    correct_array, confidence_array = _validate_prediction_inputs(correct, confidences)
    _validate_bin_count(n_bins)
    rows = []

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins

        if bin_index == 0:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array > lower) & (confidence_array <= upper)

        count = int(in_bin.sum())

        # keep empty bins so calibration output always has the configured shape
        if count == 0:
            rows.append({
                "bin": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": 0,
                "bin_accuracy": None,
                "bin_confidence": None
            })
            continue

        rows.append({
            "bin": bin_index,
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "bin_accuracy": float(correct_array[in_bin].mean()),
            "bin_confidence": float(confidence_array[in_bin].mean())
        })

    return rows