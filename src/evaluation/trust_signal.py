"""Rule-based trust signal.

This is an experimental warning signal, not a safety guarantee.
It compares degraded results against the undegraded baseline.
"""

VALID_SIGNALS = {"trust", "caution", "do_not_trust"}

def assign_trust_signal(condition_metrics: dict, clean_metrics: dict) -> dict:
    # measure how much each metric has changed from the undegraded baseline
    accuracy_drop = clean_metrics["accuracy"] - condition_metrics["accuracy"]
    ece_increase = condition_metrics["ece"] - clean_metrics["ece"]
    gap_increase = (condition_metrics["confidence_accuracy_gap"] - clean_metrics["confidence_accuracy_gap"])
    hcer = condition_metrics["hcer"]

    # provisional thresholds chosen for initial testing, will be refined from experiment results    
    if (
        accuracy_drop > 0.15
        or ece_increase > 0.08
        or gap_increase > 0.12
        or hcer > 0.07
    ):
        signal = "do_not_trust"
    elif (
        accuracy_drop > 0.05
        or ece_increase > 0.03
        or gap_increase > 0.05
        or hcer > 0.02
    ):
        signal = "caution"
    else:
        signal = "trust"

    return {
        "dataset": condition_metrics["dataset"],
        "model": condition_metrics["model"],
        "degradation": condition_metrics["degradation"],
        "severity": condition_metrics["severity"],
        "trust_signal": signal,
        "accuracy_drop": accuracy_drop,
        "ece_increase": ece_increase,
        "gap_increase": gap_increase,
        "hcer": hcer,
    }