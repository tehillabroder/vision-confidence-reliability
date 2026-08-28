"""Tests for trust-signal output creation."""

import pandas as pd
import pytest

from scripts.add_trust_signal import build_trust_records
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

def metrics_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "none",
            "severity": 0,
            "accuracy": 0.98,
            "ece": 0.02,
            "confidence_accuracy_gap": 0.01,
            "hcer_fixed": 0.005,
            "hcer_adaptive": 0.002
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "blur",
            "severity": 1,
            "accuracy": 0.95,
            "ece": 0.02,
            "confidence_accuracy_gap": 0.01,
            "hcer_fixed": 0.005,
            "hcer_adaptive": 0.002
        }
    ])

def test_build_trust_records_uses_one_shared_baseline():
    # confirm every condition is compared with the same undegraded row
    records = build_trust_records(metrics_frame(), TRUST_POLICY)

    assert len(records) == 2
    assert records[0]["trust_signal"] == "trust"
    assert records[0]["gap_direction"] == "unchanged"
    assert records[0]["triggered_rule_explanations"] == []
    assert records[1]["trust_signal"] == "caution"
    assert records[1]["performance_signal"] == "caution"
    assert records[1]["confidence_signal"] == "trust"
    assert records[1]["triggered_rules"] == ["caution_relative_error_increase"]
    assert len(records[1]["triggered_rule_explanations"]) == 1

def test_build_trust_records_rejects_missing_hcer_column():
    # ensure old metric files do not silently miss out adaptive HCER
    metrics_df = metrics_frame().drop(columns=["hcer_adaptive"])

    with pytest.raises(ValueError, match="hcer_adaptive"):
        build_trust_records(metrics_df, TRUST_POLICY)