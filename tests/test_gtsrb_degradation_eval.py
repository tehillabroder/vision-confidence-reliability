"""Tests for GTSRB degradation evaluation."""

import json
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from experiments.gtsrb_degradation_eval import (
    build_experiment_conditions,
    evaluate_condition,
    load_validation_profile,
    save_evaluation_outputs,
    summarise_condition
)

class StaticModel(nn.Module):
    """Return fixed predictions for synthetic images."""
    
    def __init__(self, predictions: torch.Tensor):
        super().__init__()
        logits = torch.full((len(predictions), 43), -4.0)
        logits[torch.arange(len(predictions)), predictions] = 4.0
        self.register_buffer("logits", logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.logits[:images.size(0)]

def valid_config() -> dict:
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "checkpoint": "checkpoints/gtsrb_cnn.pt",
        "seed": 42,
        "evaluation": {
            "fixed_hcer_threshold": 0.90,
            "adaptive_hcer_percentile": 90
        }
    }

def valid_profile() -> dict:
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "checkpoint": "checkpoints/gtsrb_cnn.pt",
        "seed": 42,
        "degradation": "none",
        "severity": 0,
        "validation_sample_count": 4000,
        "fixed_hcer_threshold": 0.90,
        "adaptive_hcer_percentile": 90,
        "adaptive_hcer_threshold": 0.80
    }

def test_build_experiment_conditions_includes_undegraded_baseline():
    # confirm the undegraded condition appears once before degraded conditions
    conditions = build_experiment_conditions(
        ["blur", "noise"],
        [1, 2]
    )
    assert conditions == [
        ("none", 0),
        ("blur", 1),
        ("blur", 2),
        ("noise", 1),
        ("noise", 2)
    ]

def test_evaluate_condition_returns_prediction_rows(monkeypatch):
    # confirm each evaluated image produces a complete prediction row
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 2, 2, 1])
    image_ids = torch.tensor([10, 11, 12, 13])
    dataset = TensorDataset(images, labels, image_ids)
    captured = {}

    def fake_dataset_builder(data_dir, degradation, severity):
        captured["data_dir"] = data_dir
        captured["degradation"] = degradation
        captured["severity"] = severity
        return dataset

    monkeypatch.setattr(
        "experiments.gtsrb_degradation_eval.build_gtsrb_test_dataset",
        fake_dataset_builder
    )
    model = StaticModel(torch.tensor([0, 1, 2, 3]))

    rows = evaluate_condition(
        model=model,
        device=torch.device("cpu"),
        data_dir="data",
        batch_size=4,
        degradation="noise",
        severity=3,
        seed=42,
        max_eval_batches=None
    )
    assert len(rows) == 4
    assert [row["correct"] for row in rows] == [1, 0, 1, 0]
    assert [row["image_id"] for row in rows] == [10, 11, 12, 13]
    assert all(row["dataset"] == "GTSRB" for row in rows)
    assert all(row["model"] == "GTSRBCNN" for row in rows)
    assert all(0 <= row["confidence"] <= 1 for row in rows)
    assert captured == {
        "data_dir": "data",
        "degradation": "noise",
        "severity": 3
    }

def test_evaluate_condition_respects_batch_limit(monkeypatch):
    # create six simple test images, which would normally make three batches
    images = torch.zeros(6, 3, 8, 8)
    labels = torch.zeros(6, dtype=torch.long)
    image_ids = torch.arange(6)
    dataset = TensorDataset(images, labels, image_ids)
    # use the small test dataset instead of loading the real GTSRB data
    # so this test only checks whether evaluation stops after one batch
    monkeypatch.setattr(
        "experiments.gtsrb_degradation_eval.build_gtsrb_test_dataset",
        lambda data_dir, degradation, severity: dataset
    )
    # one batch contains two images, so the model only needs two predictions
    model = StaticModel(torch.zeros(2, dtype=torch.long))

    rows = evaluate_condition(
        model=model,
        device=torch.device("cpu"),
        data_dir="data",
        batch_size=2,
        degradation="none",
        severity=0,
        seed=42,
        max_eval_batches=1
    )
    # confirm evaluation stopped after the first batch rather than processing all six images
    assert len(rows) == 2

def test_evaluate_condition_rejects_empty_dataset(monkeypatch):
    # create an empty dataset to represent missing evaluation data
    images = torch.empty(0, 3, 8, 8)
    labels = torch.empty(0, dtype=torch.long)
    image_ids = torch.empty(0, dtype=torch.long)
    dataset = TensorDataset(images, labels, image_ids)

    # replace real GTSRB loading so the function receives no evaluation samples
    monkeypatch.setattr(
        "experiments.gtsrb_degradation_eval.build_gtsrb_test_dataset",
        lambda data_dir, degradation, severity: dataset
    )
    # no predictions are needed because the loader contains no batches
    model = StaticModel(torch.empty(0, dtype=torch.long))
    
    # confirm the function rejects an empty run instead of returning empty results
    with pytest.raises(ValueError, match="Evaluation produced no predictions"):
        evaluate_condition(
            model=model,
            device=torch.device("cpu"),
            data_dir="data",
            batch_size=2,
            degradation="none",
            severity=0,
            seed=42,
            max_eval_batches=None
        )

def test_summarise_condition_includes_balanced_accuracy_and_hcer():
    # confirm GTSRB summaries include class balance and both HCER variants
    rows = [
        {
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "seed": 42,
            "degradation": "noise",
            "severity": 5,
            "true_label": 0,
            "predicted_label": 0,
            "correct": 1,
            "confidence": 0.95
        },
        {
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "seed": 42,
            "degradation": "noise",
            "severity": 5,
            "true_label": 0,
            "predicted_label": 1,
            "correct": 0,
            "confidence": 0.85
        },
        {
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "seed": 42,
            "degradation": "noise",
            "severity": 5,
            "true_label": 1,
            "predicted_label": 1,
            "correct": 1,
            "confidence": 0.80
        },
        {
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "seed": 42,
            "degradation": "noise",
            "severity": 5,
            "true_label": 1,
            "predicted_label": 0,
            "correct": 0,
            "confidence": 0.75
        }
    ]

    summary = summarise_condition(
        rows=rows,
        ece_bins=10,
        fixed_hcer_threshold=0.90,
        adaptive_hcer_threshold=0.80,
        adaptive_hcer_percentile=90
    )

    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["balanced_accuracy"] == pytest.approx(0.5)
    # only the 0.95 prediction meets the fixed 0.90 threshold and it is correct
    assert summary["hcer"] == pytest.approx(0.0)
    assert summary["hcer_fixed"] == pytest.approx(0.0)
    # the adaptive 0.80 threshold includes three predictions, with one high-confidence error
    assert summary["hcer_adaptive"] == pytest.approx(0.25)
    assert summary["num_examples"] == 4

def test_load_validation_profile_accepts_matching_profile(tmp_path):
    # confirm matching validation evidence can be loaded
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(valid_profile()),
        encoding="utf-8"
    )

    profile = load_validation_profile(profile_path, valid_config())

    assert profile["adaptive_hcer_threshold"] == 0.80

@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("dataset", "MNIST"),
        ("model", "SimpleCNN"),
        ("checkpoint", "checkpoints/other.pt"),
        ("seed", 7),
        ("fixed_hcer_threshold", 0.75),
        ("adaptive_hcer_percentile", 95)
    ]
)
def test_load_validation_profile_rejects_mismatch(tmp_path, name, value):
    # ensure evaluation cannot use unrelated validation evidence
    profile = valid_profile()
    profile[name] = value
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(profile),
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match=f"Validation profile {name}"):
        load_validation_profile(profile_path, valid_config())

def test_save_evaluation_outputs_creates_expected_files(tmp_path):
    # check that every required evaluation evidence file is saved
    config_path = tmp_path / "gtsrb.yaml"
    config_path.write_text("dataset: GTSRB\n", encoding="utf-8")
    output_dir = tmp_path / "results"

    paths = save_evaluation_outputs(
        prediction_rows=[{
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "image_id": 0,
            "correct": 1,
            "confidence": 0.9
        }],
        metric_rows=[{
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "accuracy": 1.0
        }],
        calibration_rows=[{
            "dataset": "GTSRB",
            "model": "GTSRBCNN",
            "bin": 9,
            "count": 1
        }],
        config_path=config_path,
        output_dir=output_dir
    )

    assert all(path.exists() for path in paths.values())
    assert paths["config"].read_text(encoding="utf-8") == "dataset: GTSRB\n"
    assert len(pd.read_csv(paths["predictions"])) == 1
    assert len(pd.read_csv(paths["metrics"])) == 1
    assert len(pd.read_csv(paths["calibration"])) == 1