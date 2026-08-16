"""Reliability metrics for confidence evaluation."""

import numpy as np

def expected_calibration_error(correct, confidences, n_bins: int = 10):
    # convert to float arrays to avoid integer division issues
    correct_array = np.asarray(correct, dtype=float)
    confidence_array = np.asarray(confidences, dtype=float)
    if correct_array.size == 0:
        raise ValueError("Cannot calculate ECE from empty inputs.")
    if correct_array.size != confidence_array.size:
        raise ValueError("Correct and confidence arrays must have the same length.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")
    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        # avoid double counting or missing exact edge values between bins        
        if bin_index == 0:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array > lower) & (confidence_array <= upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = correct_array[in_bin].mean()
        bin_confidence = confidence_array[in_bin].mean()
        bin_weight = in_bin.mean()
        # penalise overconfidence and underconfidence equally
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return float(ece)

def high_confidence_error_rate(correct, confidences, threshold: float = 0.90):
    # calculates the share of all predictions that are wrong but highly confident
    correct_array = np.asarray(correct, dtype=int)
    confidence_array = np.asarray(confidences, dtype=float)
    if correct_array.size == 0:
        raise ValueError("Cannot calculate HCER from empty inputs.")
    if correct_array.size != confidence_array.size:
        raise ValueError("Correct and confidence arrays must have the same length.")
    high_confidence_wrong = (correct_array == 0) & (confidence_array >= threshold)
    # denominator is total predictions (not just high conf ones) to measure global risk severity
    return float(high_confidence_wrong.mean())

def high_confidence_coverage(confidences, threshold: float = 0.90):
    confidence_array = np.asarray(confidences, dtype=float)
    if confidence_array.size == 0:
        raise ValueError("Cannot calculate high-confidence coverage from empty inputs.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    return float((confidence_array >= threshold).mean())

def _rank_based_selection_mask(confidences, top_fraction: float) -> np.ndarray:
    confidence_array = np.asarray(confidences, dtype=float)
    if confidence_array.size == 0:
        raise ValueError("Cannot rank empty confidence inputs.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("Top confidence fraction must be greater than 0 and at most 1.")

    selected_count = max(1, int(np.ceil(confidence_array.size * top_fraction)))
    # stable sorting makes equal-confidence selection reproducible
    ranked_indices = np.argsort(-confidence_array, kind="stable")
    selected = np.zeros(confidence_array.size, dtype=bool)
    selected[ranked_indices[:selected_count]] = True
    return selected

def rank_based_high_confidence_error_rate(correct, confidences, top_fraction: float = 0.10):
    correct_array = np.asarray(correct, dtype=int)
    if correct_array.size == 0:
        raise ValueError("Cannot calculate rank-based HCER from empty inputs.")
    if correct_array.size != len(confidences):
        raise ValueError("Correct and confidence arrays must have the same length.")

    selected = _rank_based_selection_mask(confidences, top_fraction)
    high_confidence_wrong = (correct_array == 0) & selected
    # keep all predictions as the denominator so this remains comparable with HCER
    return float(high_confidence_wrong.mean())

def rank_based_high_confidence_coverage(confidences, top_fraction: float = 0.10):
    selected = _rank_based_selection_mask(confidences, top_fraction)
    return float(selected.mean())

def calibration_bins(correct, confidences, n_bins: int = 10):
    # groups predictions into bins to allow for plotting calibration curves
    correct_array = np.asarray(correct, dtype=float)
    confidence_array = np.asarray(confidences, dtype=float)
    rows = []
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == 0:
            in_bin = (confidence_array >= lower) & (confidence_array <= upper)
        else:
            in_bin = (confidence_array > lower) & (confidence_array <= upper)
        count = int(in_bin.sum())
        # explicitly include empty bins to prevent missing steps or gaps on the reliability plot
        if count == 0:
            rows.append({
                "bin": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": 0,
                "bin_accuracy": None,
                "bin_confidence": None,
            })
            continue
        rows.append({
            "bin": bin_index,
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "bin_accuracy": float(correct_array[in_bin].mean()),
            "bin_confidence": float(confidence_array[in_bin].mean()),
        })
    return rows