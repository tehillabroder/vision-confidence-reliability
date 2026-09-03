"""Tests for GTSRB model comparison reporting."""

import pandas as pd
import pytest
from src.reporting.model_comparison import build_confirmatory_model_summary, build_model_comparison, build_trust_transition_comparison, save_model_comparison_plots

# build lightweight mock metrics and trust records to simulate a baseline vs stronger run
def build_metrics_frame(model: str, accuracy_shift: float = 0.0, ece_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    conditions = [("none", 0), ("blur", 1), ("noise", 1), ("low_light", 1)]

    for index, (degradation, severity) in enumerate(conditions):
        rows.append({
            "dataset": "GTSRB",
            "model": model,
            "seed": 42,
            "degradation": degradation,
            "severity": severity,
            "validation_split": "stratified_track",
            "requested_validation_size": 4000,
            "validation_size": 3990,
            "validation_track_count": 133,
            "track_overlap": 0,
            "validation_track_hash": "a" * 64,
            "accuracy": 0.90 - index * 0.05 + accuracy_shift,
            "balanced_accuracy": 0.88 - index * 0.05 + accuracy_shift,
            "mean_confidence": 0.92 - index * 0.03,
            "confidence_accuracy_gap": 0.02 + index * 0.02,
            "ece": 0.03 + index * 0.02 + ece_shift,
            "hcer_fixed": 0.02 + index * 0.01,
            "fixed_hcer_threshold": 0.90,
            "rank_hcer_top_fraction": 0.10,
            "adaptive_hcer_percentile": 90,
            "ece_bins": 10,
            "num_examples": 12630
        })

    return pd.DataFrame(rows)

def build_trust_records(model: str, blur_do_not_trust: int) -> list[dict]:
    records = [{"dataset": "GTSRB", "model": model, "degradation": "none", "severity": 0, "trust_signal": "trust"}]

    for severity in range(1, 6):
        if severity < 3:
            blur_signal = "trust"
        elif severity < blur_do_not_trust:
            blur_signal = "caution"
        else:
            blur_signal = "do_not_trust"

        noise_signal = "caution" if severity == 1 else "do_not_trust"

        if severity < 4:
            low_light_signal = "trust"
        elif severity == 4:
            low_light_signal = "caution"
        else:
            low_light_signal = "do_not_trust"

        records.extend([
            {"dataset": "GTSRB", "model": model, "degradation": "blur", "severity": severity, "trust_signal": blur_signal},
            {"dataset": "GTSRB", "model": model, "degradation": "noise", "severity": severity, "trust_signal": noise_signal},
            {"dataset": "GTSRB", "model": model, "degradation": "low_light", "severity": severity, "trust_signal": low_light_signal}
        ])

    return records

def build_confirmatory_metrics_frame(model: str) -> pd.DataFrame:
    rows = []

    for degradation, severity, accuracy in [
        ("none", 0, 0.75),
        ("noise", 1, 0.75),
        ("noise", 2, 0.50),
        ("noise", 3, 0.50),
        ("noise", 4, 0.25),
        ("noise", 5, 0.25)
    ]:
        rows.append({
            "dataset": "GTSRB",
            "model": model,
            "seed": 42,
            "degradation": degradation,
            "severity": severity,
            "validation_split": "stratified_track",
            "requested_validation_size": 4000,
            "validation_size": 3990,
            "validation_track_count": 133,
            "track_overlap": 0,
            "validation_track_hash": "a" * 64,
            "accuracy": accuracy,
            "balanced_accuracy": accuracy,
            "mean_confidence": 0.60,
            "confidence_accuracy_gap": 0.0,
            "ece": 0.0,
            "hcer_fixed": 0.0,
            "hcer_rank": 0.0,
            "hcer_rank_coverage": 0.25,
            "fixed_hcer_threshold": 0.90,
            "rank_hcer_top_fraction": 0.10,
            "adaptive_hcer_percentile": 90,
            "ece_bins": 10,
            "num_examples": 4
        })

    return pd.DataFrame(rows)

def build_confirmatory_predictions(model: str) -> pd.DataFrame:
    rows = []
    conditions = [
        ("none", 0, [1, 1, 1, 0], [0.90, 0.80, 0.70, 0.60]),
        ("noise", 1, [1, 1, 1, 0], [0.90, 0.80, 0.70, 0.60]),
        ("noise", 2, [1, 1, 0, 0], [0.90, 0.80, 0.40, 0.30]),
        ("noise", 3, [1, 1, 0, 0], [0.90, 0.75, 0.35, 0.25]),
        ("noise", 4, [1, 0, 0, 0], [0.90, 0.40, 0.30, 0.20]),
        ("noise", 5, [1, 0, 0, 0], [0.85, 0.50, 0.35, 0.10])
    ]
    true_labels = [0, 0, 1, 1]

    for degradation, severity, correct, confidences in conditions:
        for image_id, true_label in enumerate(true_labels):
            rows.append({
                "dataset": "GTSRB",
                "model": model,
                "seed": 42,
                "image_id": image_id,
                "true_label": true_label,
                "predicted_label": true_label if correct[image_id] else 1 - true_label,
                "correct": correct[image_id],
                "confidence": confidences[image_id],
                "degradation": degradation,
                "severity": severity
            })

    return pd.DataFrame(rows)

def build_confirmatory_trust_records(model: str) -> list[dict]:
    return [
        {"model": model, "degradation": "none", "severity": 0, "trust_signal": "trust", "performance_signal": "trust", "confidence_signal": "trust"},
        {"model": model, "degradation": "blur", "severity": 1, "trust_signal": "trust", "performance_signal": "trust", "confidence_signal": "trust"},
        {"model": model, "degradation": "blur", "severity": 3, "trust_signal": "caution", "performance_signal": "caution", "confidence_signal": "trust"},
        {"model": model, "degradation": "noise", "severity": 1, "trust_signal": "do_not_trust", "performance_signal": "do_not_trust", "confidence_signal": "do_not_trust"},
        {"model": model, "degradation": "low_light", "severity": 1, "trust_signal": "trust", "performance_signal": "trust", "confidence_signal": "trust"},
        {"model": model, "degradation": "low_light", "severity": 5, "trust_signal": "do_not_trust", "performance_signal": "do_not_trust", "confidence_signal": "trust"}
    ]

# test the core comparison logic
# checks pass when splits match, fail when conditions drift,
# and track when warning flags (caution / do-not-trust) first kick in
def test_build_model_comparison_joins_matching_conditions():
    # confirm paired rows retain both models and a signed metric difference
    first = build_metrics_frame("GTSRBCNN")
    second = build_metrics_frame("ResNet18", accuracy_shift=0.05, ece_shift=-0.01)

    comparison = build_model_comparison(first, second)

    assert len(comparison) == 4
    assert comparison.loc[0, "accuracy_delta_ResNet18_minus_GTSRBCNN"] == pytest.approx(0.05)
    assert comparison.loc[0, "ece_delta_ResNet18_minus_GTSRBCNN"] == pytest.approx(-0.01)

def test_build_model_comparison_rejects_different_conditions():
    # ensure experiments cannot be compared if their condition grids differ
    first = build_metrics_frame("GTSRBCNN")
    second = build_metrics_frame("ResNet18").iloc[:-1].copy()

    with pytest.raises(ValueError, match="identical degradation conditions"):
        build_model_comparison(first, second)

def test_build_model_comparison_rejects_different_split_hash():
    # ensure results from different track evidence cannot be mixed
    first = build_metrics_frame("GTSRBCNN")
    second = build_metrics_frame("ResNet18")
    second["validation_track_hash"] = "b" * 64

    with pytest.raises(ValueError, match="matching validation_track_hash"):
        build_model_comparison(first, second)

def test_build_trust_transition_comparison_records_first_warning_levels():
    # confirm the table records the first caution and do not trust severity
    first = build_trust_records("GTSRBCNN", blur_do_not_trust=4)
    second = build_trust_records("ResNet18", blur_do_not_trust=5)

    comparison = build_trust_transition_comparison(first, second)
    blur = comparison[comparison["degradation"] == "blur"].iloc[0]

    assert blur["GTSRBCNN_first_caution_severity"] == 3
    assert blur["GTSRBCNN_first_do_not_trust_severity"] == 4
    assert blur["ResNet18_first_caution_severity"] == 3
    assert blur["ResNet18_first_do_not_trust_severity"] == 5

def test_save_model_comparison_plots_creates_expected_files(tmp_path):
    # check that each core comparison metric produces one figure
    first = build_metrics_frame("GTSRBCNN")
    second = build_metrics_frame("ResNet18", accuracy_shift=0.05, ece_shift=-0.01)

    paths = save_model_comparison_plots(first, second, tmp_path)

    expected = {
        "accuracy_model_comparison.png",
        "balanced_accuracy_model_comparison.png",
        "mean_confidence_model_comparison.png",
        "confidence_accuracy_gap_model_comparison.png",
        "ece_model_comparison.png",
        "hcer_fixed_model_comparison.png"
    }

    assert {path.name for path in paths} == expected
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

def test_build_confirmatory_model_summary_keeps_only_selected_findings():
    # check the confirmatory table preserves the severe-noise and warning evidence
    evidence = []

    for model in ("GTSRBCNN", "ResNet18", "MobileNetV2"):
        evidence.append((
            build_confirmatory_metrics_frame(model),
            build_confirmatory_predictions(model),
            build_confirmatory_trust_records(model)
        ))

    result = build_confirmatory_model_summary(evidence)

    assert result["model"].tolist() == ["GTSRBCNN", "ResNet18", "MobileNetV2"]
    assert result["clean_test_accuracy"].eq(0.75).all()
    assert result["noise_5_accuracy"].eq(0.25).all()
    assert result["noise_5_failure_detection_auroc"].eq(1.0).all()
    assert result["noise_5_rank_high_confidence_accuracy"].eq(1.0).all()
    assert result["noise_4_to_5_persistent_error_count"].eq(3).all()
    # convert to lists so pytest.approx handles float tolerance instead of strict equality
    assert result["noise_4_to_5_persistent_error_confidence_increased_rate"].tolist() == pytest.approx([2 / 3] * 3)
    assert result["noise_4_to_5_persistent_error_mean_confidence_change"].tolist() == pytest.approx([1 / 60] * 3)
    assert result["blur_first_warning"].eq("caution@3 (performance)").all()
    assert result["noise_first_warning"].eq("do_not_trust@1 (performance+confidence)").all()
    assert result["low_light_first_warning"].eq("do_not_trust@5 (performance)").all()