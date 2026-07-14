"""Tests for reliability plot generation."""

import pandas as pd
import pytest

from src.reporting.plots import save_reliability_plots

def build_metrics_frame() -> pd.DataFrame:
    # small metrics summary covering each degradation type
    return pd.DataFrame([
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "none",
            "severity": 0,
            "accuracy": 0.98,
            "mean_confidence": 0.97,
            "confidence_accuracy_gap": -0.01,
            "ece": 0.01,
            "hcer": 0.01
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "blur",
            "severity": 1,
            "accuracy": 0.90,
            "mean_confidence": 0.85,
            "confidence_accuracy_gap": -0.05,
            "ece": 0.05,
            "hcer": 0.02
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "noise",
            "severity": 1,
            "accuracy": 0.88,
            "mean_confidence": 0.91,
            "confidence_accuracy_gap": 0.03,
            "ece": 0.04,
            "hcer": 0.03
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "degradation": "low_light",
            "severity": 1,
            "accuracy": 0.95,
            "mean_confidence": 0.80,
            "confidence_accuracy_gap": -0.15,
            "ece": 0.15,
            "hcer": 0.01
        }
    ])

def test_save_reliability_plots_creates_expected_files(tmp_path):
    # check that every expected plot file is created
    metrics_df = build_metrics_frame()
    saved_paths = save_reliability_plots(metrics_df, tmp_path)

    expected_names = {
        "blur_accuracy_and_confidence.png",
        "noise_accuracy_and_confidence.png",
        "low_light_accuracy_and_confidence.png",
        "ece_by_severity.png",
        "confidence_accuracy_gap_by_severity.png",
        "hcer_by_severity.png"
    }

    assert {path.name for path in saved_paths} == expected_names
    assert all(path.exists() for path in saved_paths)
    assert all(path.stat().st_size > 0 for path in saved_paths)

def test_save_reliability_plots_rejects_missing_columns(tmp_path):
    # check that missing metric data raises an error
    metrics_df = build_metrics_frame().drop(columns=["ece"])

    with pytest.raises(ValueError, match="Missing required metric columns"):
        save_reliability_plots(metrics_df, tmp_path)