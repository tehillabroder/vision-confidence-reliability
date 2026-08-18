"""Tests for shared degradation evaluation helpers."""

import pandas as pd
import pytest
import torch
import torch.nn as nn
from src.evaluation.runner import build_calibration_rows, build_experiment_conditions, collect_prediction_rows, save_core_evaluation_outputs, validate_evaluation_settings

class StaticModel(nn.Module):
    """Return fixed predictions for synthetic images."""

    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        logits = torch.full((len(predictions), 2), -4.0)
        logits[torch.arange(len(predictions)), predictions] = 4.0
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits[:images.size(0)]

def test_build_experiment_conditions_includes_clean_baseline_once():
    # confirm the clean condition appears once before degraded conditions
    conditions = build_experiment_conditions(["blur", "noise"], [1, 2])
    assert conditions == [
        ("none", 0),
        ("blur", 1),
        ("blur", 2),
        ("noise", 1),
        ("noise", 2)
    ]

def test_collect_prediction_rows_saves_predictions_and_respects_batch_limit():
    # check that shared inference keeps prediction metadata and honours the batch limit
    loader = [
        (
            torch.zeros(2, 1, 4, 4),
            torch.tensor([0, 1]),
            torch.tensor([10, 11])
        ),
        (
            torch.zeros(2, 1, 4, 4),
            torch.tensor([1, 0]),
            torch.tensor([12, 13])
        )
    ]
    model = StaticModel(torch.tensor([0, 0]))

    rows = collect_prediction_rows(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        dataset_name="Example",
        model_name="StaticModel",
        degradation="noise",
        severity=3,
        seed=42,
        max_eval_batches=1
    )

    assert len(rows) == 2
    assert [row["image_id"] for row in rows] == [10, 11]
    assert [row["correct"] for row in rows] == [1, 0]
    assert all(row["dataset"] == "Example" for row in rows)
    assert all(row["model"] == "StaticModel" for row in rows)
    assert all(row["degradation"] == "noise" for row in rows)
    assert all(row["severity"] == 3 for row in rows)
    assert all(0 <= row["confidence"] <= 1 for row in rows)

def test_collect_prediction_rows_rejects_empty_loader():
    # ensure empty evaluation data cannot silently produce empty outputs
    model = StaticModel(torch.empty(0, dtype=torch.long))

    with pytest.raises(ValueError, match="Evaluation produced no predictions"):
        collect_prediction_rows(
            model=model,
            loader=[],
            device=torch.device("cpu"),
            dataset_name="Example",
            model_name="StaticModel",
            degradation="none",
            severity=0,
            seed=42,
            max_eval_batches=None
        )

def test_build_calibration_rows_produces_consistent_schema():
    # tests that calibration evidence uses one shared column order
    rows = [
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "seed": 42,
            "degradation": "blur",
            "severity": 2,
            "correct": 1,
            "confidence": 0.90
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "seed": 42,
            "degradation": "blur",
            "severity": 2,
            "correct": 0,
            "confidence": 0.60
        }
    ]
    calibration_rows = build_calibration_rows(rows, ece_bins=2)

    assert len(calibration_rows) == 2
    assert list(calibration_rows[0]) == [
        "bin",
        "bin_lower",
        "bin_upper",
        "count",
        "bin_accuracy",
        "bin_confidence",
        "dataset",
        "model",
        "seed",
        "degradation",
        "severity",
        "ece_bins"
    ]

def test_build_calibration_rows_rejects_empty_predictions():
    # ensure calibration evidence cannot be built without predictions
    with pytest.raises(ValueError, match="empty predictions"):
        build_calibration_rows([], ece_bins=10)

@pytest.mark.parametrize(
    ("ece_bins", "fixed_hcer_threshold", "adaptive_hcer_threshold"),
    [
        (0, 0.90, 0.80),
        (10, -0.01, 0.80),
        (10, 0.90, 1.01)
    ]
)
def test_validate_evaluation_settings_rejects_invalid_values(ece_bins, fixed_hcer_threshold, adaptive_hcer_threshold):
    # ensure shared reliability settings remain within valid ranges
    with pytest.raises(ValueError):
        validate_evaluation_settings(ece_bins, fixed_hcer_threshold, adaptive_hcer_threshold)

def test_save_core_evaluation_outputs_creates_expected_files(tmp_path):
    # confirm the shared evaluation evidence files are saved
    # GTSRB already has a unit test that makes sure split_metadata.json has been saved
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: Example\n", encoding="utf-8")

    paths = save_core_evaluation_outputs(
        prediction_rows=[{"dataset": "Example", "confidence": 0.9}],
        metric_rows=[{"dataset": "Example", "accuracy": 1.0}],
        calibration_rows=[{"dataset": "Example", "bin": 9, "count": 1}],
        config_path=config_path,
        output_dir=tmp_path / "results"
    )
    assert set(paths) == {"predictions", "metrics", "calibration", "config"}
    assert all(path.exists() for path in paths.values())
    assert paths["config"].read_text(encoding="utf-8") == "dataset: Example\n"
    assert len(pd.read_csv(paths["predictions"])) == 1
    assert len(pd.read_csv(paths["metrics"])) == 1
    assert len(pd.read_csv(paths["calibration"])) == 1

@pytest.mark.parametrize(
    ("prediction_rows", "metric_rows", "calibration_rows"),
    # 3 scenarios where part of required file's data is empty:
    # Test 1: Missing prediction_rows throws error.
    # Test 2: Missing metric_rows throws error.
    # Test 3: Missing calibration_rows hrows error.
    [
        ([], [{"accuracy": 1.0}], [{"bin": 9}]),
        ([{"confidence": 0.9}], [], [{"bin": 9}]),
        ([{"confidence": 0.9}], [{"accuracy": 1.0}], [])
    ]
)
def test_save_core_evaluation_outputs_rejects_empty_evidence(tmp_path, prediction_rows, metric_rows, calibration_rows):
    # ensure incomplete evaluation evidence is never saved
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset: Example\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not be empty"):
        save_core_evaluation_outputs(
            prediction_rows=prediction_rows,
            metric_rows=metric_rows,
            calibration_rows=calibration_rows,
            config_path=config_path,
            output_dir=tmp_path / "results"
        )