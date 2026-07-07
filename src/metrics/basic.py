"""Basic classification and confidence metrics."""

import numpy as np

def accuracy_from_correct(correct):
    # correct should contain 1 for correct predictions and 0 for errors.
    correct_array = np.asarray(correct)
    if correct_array.size == 0:
        raise ValueError("Cannot calculate accuracy from an empty list.")
    return float(correct_array.mean())

def mean_confidence(confidences):
    confidence_array = np.asarray(confidences, dtype=float)
    if confidence_array.size == 0:
        raise ValueError("Cannot calculate mean confidence from an empty list.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")
    return float(confidence_array.mean())

def confidence_accuracy_gap(correct, confidences):
    # positive values mean that confidence is higher than observed accuracy.
    return mean_confidence(confidences) - accuracy_from_correct(correct)