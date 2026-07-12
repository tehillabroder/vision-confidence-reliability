"""Unit tests for the rule-based trust signal."""

from src.evaluation.trust_signal import assign_trust_signal

# fixed example baseline used to compare small, moderate and large metric changes
UNDEGRADED_BASELINE = {
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "degradation": "none",
    "severity": 0,
    "accuracy": 0.98,
    "ece": 0.02,
    "confidence_accuracy_gap": 0.01,
    "hcer": 0.00,
}

def test_trust_signal_returns_trust_for_small_change():
    # check small changes from the undegraded baseline remain trusted
    condition = {
        **UNDEGRADED_BASELINE,
        "degradation": "blur",
        "severity": 1,
        "accuracy": 0.96,
        "ece": 0.03,
        "confidence_accuracy_gap": 0.02,
        "hcer": 0.01,
    }
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE)
    assert result["trust_signal"] == "trust"

def test_trust_signal_returns_caution_for_medium_change():
    # check moderate changes from the undegraded baseline trigger caution
    condition = {
        **UNDEGRADED_BASELINE,
        "degradation": "blur",
        "severity": 3,
        "accuracy": 0.90,
        "ece": 0.06,
        "confidence_accuracy_gap": 0.05,
        "hcer": 0.02,
    }
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE)
    assert result["trust_signal"] == "caution"

def test_trust_signal_returns_do_not_trust_for_large_change():
    # check large changes from the undegraded baseline trigger do not trust
    condition = {
        **UNDEGRADED_BASELINE,
        "degradation": "noise",
        "severity": 5,
        "accuracy": 0.70,
        "ece": 0.15,
        "confidence_accuracy_gap": 0.20,
        "hcer": 0.10,
    }
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE)
    assert result["trust_signal"] == "do_not_trust"