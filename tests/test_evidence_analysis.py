"""Tests for offline GTSRB evidence analysis."""

from __future__ import annotations
import pandas as pd
import pytest
from src.evaluation.trust_signal import assign_trust_signal
from src.reporting.evidence_analysis import (
    build_class_failure_summary, build_confidence_diagnostics, build_paired_model_failures,
    build_prediction_confidence_transitions, build_trust_rule_ablation, build_trust_rule_attribution
)

TRUST_POLICY = {
    "error_denominator_floor": 0.01,
    "active_metrics": [
        "absolute_accuracy_drop",
        "relative_error_increase",
        "ece_increase",
        "gap_deterioration",
        "fixed_hcer_increase"
    ],
    "caution": {
        "absolute_accuracy_drop": 0.05,
        "relative_error_increase": 1.0,
        "ece_increase": 0.03,
        "gap_deterioration": 0.05,
        "fixed_hcer_increase": 0.02
    },
    "do_not_trust": {
        "absolute_accuracy_drop": 0.15,
        "relative_error_increase": 3.0,
        "ece_increase": 0.08,
        "gap_deterioration": 0.12,
        "fixed_hcer_increase": 0.07
    }
}

def prediction_frame(model: str, predicted_labels: list[int], correct: list[int], confidences: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": ["GTSRB"] * 4,
        "model": [model] * 4,
        "seed": [42] * 4,
        "image_id": [0, 1, 2, 3],
        "true_label": [0, 0, 1, 1],
        "predicted_label": predicted_labels,
        "correct": correct,
        "confidence": confidences,
        "degradation": ["noise"] * 4,
        "severity": [5] * 4
    })

def trust_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "dataset": "GTSRB",
            "model": "ExampleModel",
            "seed": 42,
            "degradation": "none",
            "severity": 0,
            "accuracy": 0.98,
            "ece": 0.02,
            "confidence_accuracy_gap": 0.01,
            "hcer_fixed": 0.005,
            "hcer_adaptive": 0.002
        },
        {
            "dataset": "GTSRB",
            "model": "ExampleModel",
            "seed": 42,
            "degradation": "noise",
            "severity": 5,
            "accuracy": 0.90,
            "ece": 0.06,
            "confidence_accuracy_gap": 0.01,
            "hcer_fixed": 0.005,
            "hcer_adaptive": 0.002
        }
    ])

def test_paired_model_failures_records_image_level_overlap():
    # similar accuracy can hide different successful images, so check the paired outcomes
    baseline = prediction_frame("Baseline", [0, 0, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6])
    stronger = prediction_frame("Stronger", [0, 1, 1, 0], [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])

    result = build_paired_model_failures(baseline, stronger).iloc[0]

    assert result["both_correct"] == 1
    assert result["baseline_only_correct"] == 1
    assert result["stronger_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["correct_set_jaccard"] == pytest.approx(1 / 3)
    assert result["same_wrong_class_rate_among_both_wrong"] == pytest.approx(1.0)

def test_paired_model_failures_rejects_different_true_labels():
    # ensure two runs cannot be paired when their underlying labels differ
    baseline = prediction_frame("Baseline", [0, 0, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6])
    stronger = prediction_frame("Stronger", [0, 1, 1, 0], [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    stronger.loc[3, "true_label"] = 0

    with pytest.raises(ValueError, match="different true labels"):
        build_paired_model_failures(baseline, stronger)

def test_class_failure_summary_compares_recall_and_class_concentration():
    # check which classes hold up and where predictions get concentrated
    baseline = prediction_frame("Baseline", [0, 0, 0, 0], [1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6])
    stronger = prediction_frame("Stronger", [0, 1, 1, 0], [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])

    result = build_class_failure_summary(baseline, stronger)
    class_zero = result[result["true_label"] == 0].iloc[0]
    class_one = result[result["true_label"] == 1].iloc[0]

    assert class_zero["baseline_recall"] == pytest.approx(1.0)
    assert class_zero["stronger_recall"] == pytest.approx(0.5)
    assert class_one["baseline_recall"] == pytest.approx(0.0)
    assert class_one["stronger_recall"] == pytest.approx(0.5)
    assert class_zero["baseline_prediction_share"] == pytest.approx(1.0)
    assert class_zero["stronger_prediction_share"] == pytest.approx(0.5)

def test_confidence_diagnostics_explains_fixed_and_rank_hcer():
    # confirm threshold coverage and conditional error reconstruct the saved HCER
    predictions = prediction_frame("ExampleModel", [0, 1, 0, 1], [1, 0, 0, 1], [0.95, 0.92, 0.40, 0.80])
    metrics = pd.DataFrame([{
        "dataset": "GTSRB",
        "model": "ExampleModel",
        "seed": 42,
        "degradation": "noise",
        "severity": 5,
        "accuracy": 0.50,
        "mean_confidence": 0.7675,
        "confidence_accuracy_gap": 0.2675,
        "ece": 0.2675,
        "hcer_fixed": 0.25,
        "hcer_rank": 0.25,
        "hcer_rank_coverage": 0.50,
        "rank_hcer_top_fraction": 0.50,
        "fixed_hcer_threshold": 0.90,
        "num_examples": 4
    }])

    result = build_confidence_diagnostics(predictions, metrics).iloc[0]

    assert result["hcer_fixed_coverage"] == pytest.approx(0.50)
    assert result["hcer_fixed_conditional_error"] == pytest.approx(0.50)
    assert result["failure_detection_auroc"] == pytest.approx(0.75)
    assert result["rank_high_confidence_accuracy"] == pytest.approx(0.50)

def test_trust_rule_attribution_separates_performance_and_confidence_rules():
    # show a performance-led severe warning with a milder confidence warning
    metrics = trust_metrics_frame()
    baseline = metrics.iloc[0].to_dict()
    saved = [assign_trust_signal(row.to_dict(), baseline, TRUST_POLICY) for _, row in metrics.iterrows()]

    result = build_trust_rule_attribution(metrics, TRUST_POLICY, saved)
    degraded = result[(result["degradation"] == "noise") & (result["severity"] == 5)]
    relative_error = degraded[degraded["rule"] == "relative_error_increase"].iloc[0]
    ece = degraded[degraded["rule"] == "ece_increase"].iloc[0]

    assert relative_error["rule_signal"] == "do_not_trust"
    assert relative_error["is_overall_driver"]
    assert ece["rule_signal"] == "caution"
    assert ece["is_channel_driver"]
    assert not ece["is_overall_driver"]

def test_trust_rule_ablation_records_minimal_rule_behaviour():
    # ensure performance rules reproduce a warning that confidence rules weaken
    result = build_trust_rule_ablation(trust_metrics_frame(), TRUST_POLICY)
    degraded = result[(result["degradation"] == "noise") & (result["severity"] == 5)]
    performance = degraded[degraded["configuration"] == "performance_rules"].iloc[0]
    confidence = degraded[degraded["configuration"] == "confidence_rules"].iloc[0]
    relative_only = degraded[degraded["configuration"] == "only_relative_error_increase"].iloc[0]

    assert performance["ablated_trust_signal"] == "do_not_trust"
    assert not performance["changed_from_full"]
    assert confidence["ablated_trust_signal"] == "caution"
    assert confidence["changed_from_full"]
    assert relative_only["ablated_trust_signal"] == "do_not_trust"

def transition_prediction_frame() -> pd.DataFrame:
    rows = []
    conditions = [
        ("none", 0, [1, 1, 0, 0], [0.90, 0.80, 0.70, 0.60]),
        ("noise", 1, [1, 0, 0, 0], [0.85, 0.85, 0.75, 0.65]),
        ("noise", 2, [1, 0, 0, 0], [0.80, 0.90, 0.80, 0.70])
    ]

    for degradation, severity, correct, confidences in conditions:
        for image_id in range(4):
            rows.append({
                "dataset": "GTSRB",
                "model": "ExampleModel",
                "seed": 42,
                "image_id": image_id,
                "true_label": image_id // 2,
                "predicted_label": image_id // 2 if correct[image_id] else 9,
                "correct": correct[image_id],
                "confidence": confidences[image_id],
                "degradation": degradation,
                "severity": severity
            })

    return pd.DataFrame(rows)

def test_prediction_confidence_transitions_tracks_persistent_errors():
    # check whether the same failed images can become more confident as degradation increases
    result = build_prediction_confidence_transitions(transition_prediction_frame())
    first = result[(result["from_severity"] == 0) & (result["to_severity"] == 1)].iloc[0]

    assert len(result) == 2
    assert first["confidence_increased_rate"] == pytest.approx(0.75)
    assert first["wrong_at_both_count"] == 2
    assert first["wrong_at_both_confidence_increased_rate"] == pytest.approx(1.0)
    assert first["wrong_at_both_mean_confidence_change"] == pytest.approx(0.05)
    assert first["correct_to_wrong_count"] == 1
    assert first["correct_to_wrong_confidence_increased_rate"] == pytest.approx(1.0)

def test_prediction_confidence_transitions_requires_matching_images():
    # image-level confidence changes only mean something when adjacent severities contain the same examples
    predictions = transition_prediction_frame()
    missing_row = (
        (predictions["degradation"] == "noise")
        & (predictions["severity"] == 2)
        & (predictions["image_id"] == 3)
    )
    predictions = predictions[~missing_row]

    with pytest.raises(ValueError, match="same images and true labels"):
        build_prediction_confidence_transitions(predictions)