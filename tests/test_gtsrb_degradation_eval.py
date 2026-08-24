"""Tests for GTSRB degradation evaluation."""

import json
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from experiments.gtsrb_degradation_eval import (
    evaluate_condition, save_evaluation_outputs, summarise_condition,
    validate_evaluation_sources, validate_gtsrb_validation_profile
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
        "evaluation": {"fixed_hcer_threshold": 0.90, "adaptive_hcer_percentile": 90, "rank_hcer_top_fraction": 0.10},
        "training": {"validation_size": 4000, "validation_split": "stratified_track"}
    }

def valid_split_metadata() -> dict:
    return {
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "validation_size": 3990,
        "validation_size_difference": -10,
        "train_size": 22650,
        "track_size": 30,
        "total_track_count": 888,
        "train_track_count": 755,
        "validation_track_count": 133,
        "track_overlap": 0,
        "training_class_count": 43,
        "validation_class_count": 43,
        "validation_track_hash": "a" * 64
    }

def valid_profile() -> dict:
    return {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "checkpoint": "checkpoints/gtsrb_cnn.pt",
        "seed": 42,
        "degradation": "none",
        "severity": 0,
        "validation_sample_count": 3990,
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "validation_track_count": 133,
        "track_overlap": 0,
        "validation_track_hash": "a" * 64,
        "split_metadata": valid_split_metadata(),
        "fixed_hcer_threshold": 0.90,
        "adaptive_hcer_percentile": 90,
        "adaptive_hcer_threshold": 0.80,
        "rank_hcer_top_fraction": 0.10
    }


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
        model_name="ResNet18",
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
    assert all(row["model"] == "ResNet18" for row in rows)
    assert all(row["seed"] == 42 for row in rows)
    assert all(row["degradation"] == "noise" for row in rows)
    assert all(row["severity"] == 3 for row in rows)
    assert all(0 <= row["confidence"] <= 1 for row in rows)
    assert captured == {
        "data_dir": "data",
        "degradation": "noise",
        "severity": 3
    }

def test_summarise_condition_includes_balanced_accuracy_and_hcer():
    # confirm GTSRB summaries include class balance and all HCER diagnostics
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
        adaptive_hcer_percentile=90,
        rank_hcer_top_fraction=0.50,
        split_metadata=valid_split_metadata()
    )


    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["balanced_accuracy"] == pytest.approx(0.5)
    # only the 0.95 prediction meets the fixed 0.90 threshold and it is correct
    assert summary["hcer"] == pytest.approx(0.0)
    assert summary["hcer_fixed"] == pytest.approx(0.0)
    # the adaptive 0.80 threshold includes three predictions, with one high-confidence error
    assert summary["hcer_adaptive"] == pytest.approx(0.25)
    assert summary["hcer_adaptive_coverage"] == pytest.approx(0.75)
    assert summary["hcer_rank"] == pytest.approx(0.25)
    assert summary["hcer_rank_coverage"] == pytest.approx(0.50)
    assert summary["rank_hcer_top_fraction"] == pytest.approx(0.50)
    assert summary["num_examples"] == 4

def test_validate_gtsrb_validation_profile_accepts_matching_profile():
    # confirm GTSRB-specific reliability and track evidence is accepted
    split_metadata = validate_gtsrb_validation_profile(valid_profile(), valid_config())

    assert split_metadata == valid_split_metadata()

def test_validate_evaluation_sources_accepts_matching_sources():
    # confirm evaluation accepts checkpoint and profile evidence from the same split
    split_metadata = valid_split_metadata()
    metadata = {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "seed": 42,
        "class_count": 43,
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "validation_size": 3990,
        "track_overlap": 0,
        "validation_track_hash": "a" * 64,
        "split_metadata": split_metadata
    }

    validated_split = validate_evaluation_sources(
        metadata,
        split_metadata,
        valid_config()
    )

    assert validated_split == split_metadata

def test_validate_gtsrb_validation_profile_rejects_rank_fraction_mismatch():
    # ensure gtsrb evaluation uses the correct rank as its validation profile
    profile = valid_profile()
    profile["rank_hcer_top_fraction"] = 0.20

    with pytest.raises(ValueError, match="rank_hcer_top_fraction"):
        validate_gtsrb_validation_profile(profile, valid_config())

def test_validate_gtsrb_validation_profile_rejects_split_evidence_mismatch():
    # ensure duplicated profile fields cannot disagree with the saved track evidence
    profile = valid_profile()
    profile["validation_track_hash"] = "c" * 64

    with pytest.raises(ValueError, match="does not match its split metadata"):
        validate_gtsrb_validation_profile(profile, valid_config())

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
        split_metadata=valid_split_metadata(),
        config_path=config_path,
        output_dir=output_dir
    )
    saved_split = json.loads(paths["split_metadata"].read_text(encoding="utf-8"))
    assert saved_split == valid_split_metadata()

    assert all(path.exists() for path in paths.values())
    assert paths["config"].read_text(encoding="utf-8") == "dataset: GTSRB\n"
    assert len(pd.read_csv(paths["predictions"])) == 1
    assert len(pd.read_csv(paths["metrics"])) == 1
    assert len(pd.read_csv(paths["calibration"])) == 1

def test_validate_evaluation_sources_rejects_split_mismatch():
    # ensure evaluation cannot combine unrelated split evidence
    metadata = {
        "dataset": "GTSRB",
        "model": "GTSRBCNN",
        "seed": 42,
        "class_count": 43,
        "validation_split": "stratified_track",
        "requested_validation_size": 4000,
        "validation_size": 3990,
        "track_overlap": 0,
        "validation_track_hash": "c" * 64,
        "split_metadata": {
            **valid_split_metadata(),
            "validation_track_hash": "c" * 64
        }
    }
    with pytest.raises(ValueError, match="different GTSRB track splits"):
        validate_evaluation_sources(metadata, valid_split_metadata(), valid_config())