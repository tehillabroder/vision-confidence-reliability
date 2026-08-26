"""Basic classification and confidence metrics."""

import numpy as np

def accuracy_from_correct(correct) -> float:
    correct_array = np.asarray(correct)
    if correct_array.size == 0:
        raise ValueError("Cannot calculate accuracy from an empty list.")
    if not np.all(np.isin(correct_array, [0, 1])):
            raise ValueError("Correct values must be 0 or 1.")
    return float(correct_array.mean())

def mean_confidence(confidences) -> float:
    confidence_array = np.asarray(confidences, dtype=float)

    if confidence_array.size == 0:
        raise ValueError("Cannot calculate mean confidence from an empty list.")
    if np.any(confidence_array < 0) or np.any(confidence_array > 1):
        raise ValueError("Confidence values must be between 0 and 1.")

    return float(confidence_array.mean())

def confidence_accuracy_gap(correct, confidences) -> float:
    if np.asarray(correct).size != np.asarray(confidences).size:
        raise ValueError("Correct and confidence arrays must have the same length.")

    return mean_confidence(confidences) - accuracy_from_correct(correct)