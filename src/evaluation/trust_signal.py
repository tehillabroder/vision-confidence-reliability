"""
Create baseline-relative trust signals.
This is an experimental warning signal, not a safety guarantee.
"""

from src.utils.config import TRUST_METRICS
VALID_SIGNALS = {"trust", "caution", "do_not_trust"}

def calculate_deterioration(condition_metrics: dict, baseline_metrics: dict, error_denominator_floor: float) -> dict:
    baseline_accuracy = float(baseline_metrics["accuracy"])
    condition_accuracy = float(condition_metrics["accuracy"])
    baseline_error = 1.0 - baseline_accuracy
    condition_error = 1.0 - condition_accuracy

    # use percentage-point loss rather than dividing accuracy by baseline accuracy
    absolute_accuracy_drop = baseline_accuracy - condition_accuracy

    # error-rate growth shows when a small accuracy loss multiplies rare baseline errors
    error_denominator = max(baseline_error, error_denominator_floor)
    relative_error_increase = (condition_error - baseline_error) / error_denominator

    baseline_gap = float(baseline_metrics["confidence_accuracy_gap"])
    condition_gap = float(condition_metrics["confidence_accuracy_gap"])
    gap_shift = condition_gap - baseline_gap

    # compare gap magnitude so overconfidence and underconfidence can both worsen
    gap_deterioration = abs(condition_gap) - abs(baseline_gap)

    # compare HCER changes so existing baseline errors are not blamed on degradation
    fixed_hcer_increase = float(condition_metrics["hcer_fixed"]) - float(baseline_metrics["hcer_fixed"])
    adaptive_hcer_increase = float(condition_metrics["hcer_adaptive"]) - float(baseline_metrics["hcer_adaptive"])

    return {
        "baseline_error": baseline_error,
        "condition_error": condition_error,
        "absolute_accuracy_drop": absolute_accuracy_drop,
        "relative_error_increase": relative_error_increase,
        "ece_increase": float(condition_metrics["ece"]) - float(baseline_metrics["ece"]),
        "gap_shift": gap_shift,
        "gap_deterioration": gap_deterioration,
        "fixed_hcer_increase": fixed_hcer_increase,
        "adaptive_hcer_increase": adaptive_hcer_increase
    }

def _find_triggered_rules(deterioration: dict, thresholds: dict, signal: str) -> list[str]:
    return [
        f"{signal}_{metric}"
        for metric in TRUST_METRICS
        if deterioration[metric] >= thresholds[metric]
    ]

def assign_trust_signal(condition_metrics: dict, baseline_metrics: dict, trust_policy: dict) -> dict:

    deterioration = calculate_deterioration(condition_metrics, baseline_metrics, trust_policy["error_denominator_floor"])
    do_not_trust_rules = _find_triggered_rules(deterioration, trust_policy["do_not_trust"], "do_not_trust")

    # any severe rule takes priority so one serious failure is not hidden
    if do_not_trust_rules:
        signal = "do_not_trust"
        triggered_rules = do_not_trust_rules
    else:
        caution_rules = _find_triggered_rules(deterioration, trust_policy["caution"], "caution")
        if caution_rules:
            signal = "caution"
            triggered_rules = caution_rules
        else:
            signal = "trust"
            triggered_rules = []

    return {
        "dataset": condition_metrics["dataset"],
        "model": condition_metrics["model"],
        "degradation": condition_metrics["degradation"],
        "severity": int(condition_metrics["severity"]),
        "trust_signal": signal,
        "triggered_rules": triggered_rules,
        "hcer_fixed": float(condition_metrics["hcer_fixed"]),
        "hcer_adaptive": float(condition_metrics["hcer_adaptive"]),
        **deterioration  # unpack all deterioration metrics
    }