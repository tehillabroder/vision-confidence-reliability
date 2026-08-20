"""Tests for GTSRB model comparison reporting."""

import pandas as pd
import pytest
from src.reporting.model_comparison import build_model_comparison, build_trust_transition_comparison, save_model_comparison_plots

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