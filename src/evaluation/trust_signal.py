"""
Create baseline-relative trust signals.
This is an experimental warning signal, not a safety guarantee.
"""
from __future__ import annotations
from src.utils.config import TRUST_METRICS
RULE_LABELS = {
    "absolute_accuracy_drop": "absolute accuracy drop",
    "relative_error_increase": "relative error increase",
    "ece_increase": "ECE increase",
    "gap_deterioration": "confidence gap deterioration",
    "fixed_hcer_increase": "fixed HCER increase",
    "adaptive_hcer_increase": "adaptive HCER increase"
}

def determine_gap_direction(gap_shift: float) -> str:
    # treat tiny floating-point differences as unchanged
    if abs(gap_shift) <= 1e-12:
        return "unchanged"
    if gap_shift > 0:
        return "towards_overconfidence"
    return "towards_underconfidence"

def calculate_deterioration(condition_metrics: dict, baseline_metrics: dict, error_denominator_floor: float) -> dict:
    baseline_accuracy = float(baseline_metrics["accuracy"])
    condition_accuracy = float(condition_metrics["accuracy"])
    baseline_error = 1.0 - baseline_accuracy
    condition_error = 1.0 - condition_accuracy

    # use percentage-point loss rather than dividing accuracy by baseline accuracy
    absolute_accuracy_drop = baseline_accuracy - condition_accuracy

    # the floor prevents a near-perfect baseline creating an unstable ratio
    error_denominator = max(baseline_error, error_denominator_floor)
    relative_error_increase = (condition_error - baseline_error) / error_denominator

    baseline_gap = float(baseline_metrics["confidence_accuracy_gap"])
    condition_gap = float(condition_metrics["confidence_accuracy_gap"])
    gap_shift = condition_gap - baseline_gap

    # signed movement gives direction while magnitude shows deterioration
    gap_direction = determine_gap_direction(gap_shift)
    gap_deterioration = abs(condition_gap) - abs(baseline_gap)

    # baseline comparison avoids blaming existing confident errors on degradation
    fixed_hcer_increase = float(condition_metrics["hcer_fixed"]) - float(baseline_metrics["hcer_fixed"])
    adaptive_hcer_increase = float(condition_metrics["hcer_adaptive"]) - float(baseline_metrics["hcer_adaptive"])

    return {
        "baseline_error": baseline_error,
        "condition_error": condition_error,
        "absolute_accuracy_drop": absolute_accuracy_drop,
        "relative_error_increase": relative_error_increase,
        "ece_increase": float(condition_metrics["ece"]) - float(baseline_metrics["ece"]),
        "gap_shift": gap_shift,
        "gap_direction": gap_direction,
        "gap_deterioration": gap_deterioration,
        "fixed_hcer_increase": fixed_hcer_increase,
        "adaptive_hcer_increase": adaptive_hcer_increase
    }

def _find_triggered_metrics(deterioration: dict, thresholds: dict, active_metrics: list[str] | tuple[str, ...]) -> list[str]:
    return [
        metric for metric in active_metrics
        if deterioration[metric] >= thresholds[metric]
    ]

def _build_rule_explanations(
    deterioration: dict,
    thresholds: dict,
    signal: str,
    triggered_metrics: list[str]
) -> list[str]:
    explanations = []
    signal_label = signal.replace("_", " ")

    for metric in triggered_metrics:
        label = RULE_LABELS[metric]
        value = deterioration[metric]
        threshold = thresholds[metric]

        if metric == "gap_deterioration":
            direction = deterioration["gap_direction"].replace("_", " ")
            label = f"{label} {direction}"

        explanations.append(
            f"{label} was {value:.4f}, meeting the "
            f"{signal_label} threshold of {threshold:.4f}."
        )

    return explanations

def assign_trust_signal(condition_metrics: dict, baseline_metrics: dict, trust_policy: dict) -> dict:

    deterioration = calculate_deterioration(condition_metrics, baseline_metrics, trust_policy["error_denominator_floor"])
    active_metrics = trust_policy.get("active_metrics", TRUST_METRICS)
    do_not_trust_metrics = _find_triggered_metrics(deterioration, trust_policy["do_not_trust"], active_metrics)

    # one severe rule is enough to justify the strongest warning
    if do_not_trust_metrics:
        signal = "do_not_trust"
        triggered_metrics = do_not_trust_metrics
        triggered_thresholds = trust_policy["do_not_trust"]
    else:
        caution_metrics = _find_triggered_metrics(
            deterioration,
            trust_policy["caution"], 
            active_metrics
        )
        if caution_metrics:
            signal = "caution"
            triggered_metrics = caution_metrics
            triggered_thresholds = trust_policy["caution"]
        else:
            signal = "trust"
            triggered_metrics = []
            triggered_thresholds = {}

    # keep stable rule names for analysis and separate explanations for readers
    triggered_rules = [
        f"{signal}_{metric}"
        for metric in triggered_metrics
    ]
    triggered_rule_explanations = _build_rule_explanations(
        deterioration,
        triggered_thresholds,
        signal,
        triggered_metrics
    )

    return {
        "dataset": condition_metrics["dataset"],
        "model": condition_metrics["model"],
        "degradation": condition_metrics["degradation"],
        "severity": int(condition_metrics["severity"]),
        "trust_signal": signal,
        "triggered_rules": triggered_rules,
        "triggered_rule_explanations": triggered_rule_explanations,
        "hcer_fixed": float(condition_metrics["hcer_fixed"]),
        "hcer_adaptive": float(condition_metrics["hcer_adaptive"]),
        **deterioration
    }