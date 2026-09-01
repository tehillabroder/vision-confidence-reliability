"""Tests for final GTSRB evidence plots."""

import pandas as pd
import pytest
from src.reporting.final_evidence_plots import build_severe_noise_results_table, save_final_evidence_outputs
PROVENANCE = {
    "validation_split": "stratified_track",
    "validation_track_hash": "example_hash",
    "track_overlap": 0
}

def build_evidence_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired_failures = pd.DataFrame([{
        "dataset": "GTSRB",
        "baseline_model": "GTSRBCNN",
        "stronger_model": "ResNet18",
        "degradation": "noise",
        "severity": 5,
        "both_correct": 20,
        "baseline_only_correct": 10,
        "stronger_only_correct": 9,
        "both_wrong": 61,
        "both_wrong_same_class_count": 2,
        "same_wrong_class_rate_among_both_wrong": 2 / 61,
        **PROVENANCE
    }])

    diagnostic_rows = []
    for model in ["GTSRBCNN", "ResNet18"]:
        diagnostic_rows.append({
            "dataset": "GTSRB",
            "model": model,
            "degradation": "none",
            "severity": 0,
            "accuracy": 0.90,
            "mean_confidence": 0.95,
            "confidence_accuracy_gap": 0.05,
            "hcer_fixed": 0.02,
            "hcer_fixed_coverage": 0.80,
            "hcer_fixed_conditional_error": 0.025,
            "failure_detection_auroc": 0.90,
            "rank_hcer_top_fraction": 0.10,
            "rank_high_confidence_accuracy": 0.99,
            **PROVENANCE
        })
        for severity in range(1, 6):
            diagnostic_rows.append({
                "dataset": "GTSRB",
                "model": model,
                "degradation": "noise",
                "severity": severity,
                "accuracy": 0.90 - severity * 0.10,
                "mean_confidence": 0.95 - severity * 0.06,
                "confidence_accuracy_gap": 0.05 + severity * 0.04,
                "hcer_fixed": [0.05, 0.12, 0.19, 0.18, 0.17][severity - 1],
                "hcer_fixed_coverage": 0.80 - severity * 0.08,
                "hcer_fixed_conditional_error": 0.05 + severity * 0.08,
                "failure_detection_auroc": 0.80 if model == "GTSRBCNN" else 0.87,
                "rank_hcer_top_fraction": 0.10,
                "rank_high_confidence_accuracy": 0.86 if model == "GTSRBCNN" else 0.99,
                **PROVENANCE
            })

    trust_rows = []
    for model in ["GTSRBCNN", "ResNet18"]:
        for degradation in ["blur", "noise", "low_light"]:
            for severity in range(1, 6):
                if severity <= 2:
                    performance_signal = "trust"
                elif severity == 3:
                    performance_signal = "caution"
                else:
                    performance_signal = "do_not_trust"

                confidence_signal = "trust" if severity == 1 else "caution"
                trust_signal = performance_signal if performance_signal != "trust" else confidence_signal
                trust_rows.append({
                    "dataset": "GTSRB",
                    "model": model,
                    "degradation": degradation,
                    "severity": severity,
                    "performance_signal": performance_signal,
                    "confidence_signal": confidence_signal,
                    "trust_signal": trust_signal,
                    **PROVENANCE
                })

    transition_rows = []
    for model in ["GTSRBCNN", "ResNet18"]:
        for degradation in ["blur", "noise", "low_light"]:
            for from_severity in range(5):
                transition_rows.append({
                    "dataset": "GTSRB",
                    "model": model,
                    "degradation": degradation,
                    "from_severity": from_severity,
                    "to_severity": from_severity + 1,
                    "wrong_at_both_mean_confidence_change": 0.01 * (from_severity - 2),
                    "wrong_at_both_confidence_increased_rate": 0.70,
                    "correct_to_wrong_mean_confidence_change": -0.04 * (from_severity + 1),
                    **PROVENANCE
                })

    return paired_failures, pd.DataFrame(diagnostic_rows), pd.DataFrame(trust_rows), pd.DataFrame(transition_rows)


def test_save_final_evidence_outputs_creates_expected_files(tmp_path):
    # check that the final research figures and table are all created
    frames = build_evidence_frames()
    saved_paths = save_final_evidence_outputs(*frames, tmp_path)

    expected_names = {
        "warning_timing_and_attribution.png",
        "severe_noise_failure_profiles.png",
        "fixed_hcer_context.png",
        "confidence_changes_on_failures.png",
        "severe_noise_results.csv"
    }

    assert {path.name for path in saved_paths} == expected_names
    assert all(path.exists() for path in saved_paths)
    assert all(path.stat().st_size > 0 for path in saved_paths)

def test_save_final_evidence_outputs_rejects_missing_columns(tmp_path):
    # ensure a figure can't be built from incomplete trust evidence
    paired, diagnostics, trust, transitions = build_evidence_frames()
    trust = trust.drop(columns=["confidence_signal"])

    with pytest.raises(ValueError, match="Trust attribution is missing columns"):
        save_final_evidence_outputs(paired, diagnostics, trust, transitions, tmp_path)

def test_save_final_evidence_outputs_rejects_mixed_split_fingerprints(tmp_path):
    # paired figures only mean something when every table comes from the same validation split
    paired, diagnostics, trust, transitions = build_evidence_frames()
    transitions["validation_track_hash"] = "different_hash"

    with pytest.raises(ValueError, match="matching validation split fingerprints"):
        save_final_evidence_outputs(paired, diagnostics, trust, transitions, tmp_path)

def test_build_severe_noise_results_table_uses_severity_five():
    # confirm the final comparison table uses the intended severe-noise condition
    paired, diagnostics, _, _ = build_evidence_frames()
    result = build_severe_noise_results_table(paired, diagnostics)

    assert result["model"].tolist() == ["GTSRBCNN", "ResNet18"]
    assert result["accuracy"].tolist() == pytest.approx([0.40, 0.40])
    assert result["top_10_confidence_accuracy"].tolist() == pytest.approx([0.86, 0.99])

def test_save_final_evidence_outputs_keeps_overwrite_protection(tmp_path):
    # make sure final figures are not silently replaced
    frames = build_evidence_frames()
    save_final_evidence_outputs(*frames, tmp_path)

    with pytest.raises(FileExistsError, match="Refusing to overwrite existing output"):
        save_final_evidence_outputs(*frames, tmp_path)
