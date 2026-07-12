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