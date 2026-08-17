"""Unit tests for the configuration-driven trust signal."""

import pytest

from src.evaluation.trust_signal import (
    assign_trust_signal, 
    calculate_deterioration, 
    determine_gap_direction
)
TRUST_POLICY = {
    "error_denominator_floor": 0.01,
    "caution": {
        "absolute_accuracy_drop": 0.05,
        "relative_error_increase": 1.0,
        "ece_increase": 0.03,
        "gap_deterioration": 0.05,
        "fixed_hcer_increase": 0.02,
        "adaptive_hcer_increase": 0.02
    },
    "do_not_trust": {
        "absolute_accuracy_drop": 0.15,
        "relative_error_increase": 3.0,
        "ece_increase": 0.08,
        "gap_deterioration": 0.12,
        "fixed_hcer_increase": 0.07,
        "adaptive_hcer_increase": 0.07
    }
}

UNDEGRADED_BASELINE = {
    "dataset": "MNIST",
    "model": "SimpleCNN",
    "degradation": "none",
    "severity": 0,
    "accuracy": 0.98,
    "ece": 0.02,
    "confidence_accuracy_gap": 0.01,
    "hcer_fixed": 0.005,
    "hcer_adaptive": 0.002
}

def condition_metrics(**changes) -> dict:
    condition = {
        **UNDEGRADED_BASELINE,
        "degradation": "blur",
        "severity": 1
    }
    condition.update(changes)
    return condition

def test_trust_signal_returns_trust_for_small_change():
    # check small changes from the undegraded baseline remain trusted
    condition = condition_metrics(
        accuracy=0.97,
        ece=0.03,
        confidence_accuracy_gap=0.02,
        hcer_fixed=0.01,
        hcer_adaptive=0.006
    )
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, TRUST_POLICY)
    assert result["trust_signal"] == "trust"
    assert result["triggered_rules"] == []
    assert result["triggered_rule_explanations"] == []

def test_trust_signal_uses_relative_error_increase():
    # confirm doubled error can warn before a large accuracy drop develops
    condition = condition_metrics(accuracy=0.95)

    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, TRUST_POLICY)

    assert result["absolute_accuracy_drop"] == pytest.approx(0.03)
    assert result["relative_error_increase"] == pytest.approx(1.5)
    assert result["triggered_rules"] == ["caution_relative_error_increase"]
    assert result["triggered_rule_explanations"] == ["relative error increase was 1.5000, meeting the caution threshold of 1.0000."]

def test_trust_signal_prioritises_do_not_trust_rules():
    # ensure one severe deterioration receives the strongest warning
    condition = condition_metrics(ece=0.11)
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, TRUST_POLICY)
    assert result["trust_signal"] == "do_not_trust"
    assert result["triggered_rules"] == ["do_not_trust_ece_increase"]
    assert result["triggered_rule_explanations"] == [
        "ECE increase was 0.0900, meeting the do not trust threshold of 0.0800."
    ]

def test_trust_signal_detects_worsening_underconfidence():
    # confirm gap magnitude detects deterioration below zero
    condition = condition_metrics(confidence_accuracy_gap=-0.061)
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, TRUST_POLICY)
    assert result["gap_shift"] == pytest.approx(-0.071)
    assert result["gap_direction"] == "towards_underconfidence"
    assert result["gap_deterioration"] == pytest.approx(0.051)
    assert result["trust_signal"] == "caution"
    assert result["triggered_rules"] == ["caution_gap_deterioration"]
    assert "towards underconfidence" in result["triggered_rule_explanations"][0]

def test_trust_signal_does_not_warn_when_gap_moves_towards_zero():
    # ensure improved alignment is not treated as deterioration
    baseline = {
        **UNDEGRADED_BASELINE,
        "confidence_accuracy_gap": 0.08
    }
    condition = {
        **baseline,
        "degradation": "low_light",
        "severity": 1,
        "confidence_accuracy_gap": 0.02
    }
    result = assign_trust_signal(condition, baseline, TRUST_POLICY)
    assert result["gap_shift"] == pytest.approx(-0.06)
    assert result["gap_direction"] == "towards_underconfidence"
    assert result["gap_deterioration"] == pytest.approx(-0.06)
    assert result["trust_signal"] == "trust"


def test_relative_error_increase_uses_denominator_floor():
    # confirm a near-perfect baseline does not create an extreme ratio
    baseline = {**UNDEGRADED_BASELINE, "accuracy": 0.999}
    condition = {**baseline, "accuracy": 0.989}

    deterioration = calculate_deterioration(condition, baseline, error_denominator_floor=0.01)

    assert deterioration["baseline_error"] == pytest.approx(0.001)
    assert deterioration["condition_error"] == pytest.approx(0.011)
    assert deterioration["relative_error_increase"] == pytest.approx(1.0)

def test_trust_signal_compares_fixed_and_adaptive_hcer_separately():
    # confirm each HCER definition keeps its own deterioration rule
    condition = condition_metrics(
        hcer_fixed=0.03,
        hcer_adaptive=0.006
    )

    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, TRUST_POLICY)

    assert result["fixed_hcer_increase"] == pytest.approx(0.025)
    assert result["adaptive_hcer_increase"] == pytest.approx(0.004)
    assert result["triggered_rules"] == ["caution_fixed_hcer_increase"]

def test_gap_direction_detects_movement_towards_overconfidence():
    # confirm a positive gap shift records the correct direction
    assert determine_gap_direction(0.02) == "towards_overconfidence"

def test_gap_direction_treats_tiny_difference_as_unchanged():
    # ensure floating-point noise does not create a false direction
    assert determine_gap_direction(1e-13) == "unchanged"

def test_trust_signal_ignores_inactive_adaptive_hcer():
    # confirm that the audited adaptive HCER stays as evidence without driving the warning
    trust_policy = {
        **TRUST_POLICY,
        "active_metrics": [
            "absolute_accuracy_drop",
            "relative_error_increase",
            "ece_increase",
            "gap_deterioration",
            "fixed_hcer_increase"
        ],
        "caution": {
            metric: value
            for metric, value in TRUST_POLICY["caution"].items()
            if metric != "adaptive_hcer_increase"
        },
        "do_not_trust": {
            metric: value
            for metric, value in TRUST_POLICY["do_not_trust"].items()
            if metric != "adaptive_hcer_increase"
        }
    }
    condition = condition_metrics(hcer_adaptive=0.50)
    result = assign_trust_signal(condition, UNDEGRADED_BASELINE, trust_policy)
    assert result["adaptive_hcer_increase"] == pytest.approx(0.498)
    assert result["trust_signal"] == "trust"
    assert result["triggered_rules"] == []