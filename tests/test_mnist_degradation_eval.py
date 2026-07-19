"""Tests for the MNIST degradation evaluation."""

import pytest
import torch

import experiments.mnist_degradation_eval as evaluation
from experiments.mnist_degradation_eval import evaluate_condition, load_evaluation_model, summarise_condition, validate_evaluation_settings
from src.models.simple_cnn import SimpleCNN

@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_evaluation_rejects_invalid_hcer_threshold(threshold):
    # ensure the threshold stays between 0 and 1
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_evaluation_settings(ece_bins=10, hcer_threshold=threshold)

def test_evaluation_rejects_invalid_ece_bin_count():
    # ensure ECE uses at least one bin
    with pytest.raises(ValueError, match="greater than zero"):
        validate_evaluation_settings(ece_bins=0, hcer_threshold=0.90)

def test_load_evaluation_model_rejects_missing_checkpoint(tmp_path):
    # check that a missing checkpoint raises an error
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_evaluation_model(tmp_path / "missing.pt", torch.device("cpu"))

def test_evaluate_condition_rejects_empty_results(monkeypatch):
    # replace the real dataset and loader so the test can simulate no data
    monkeypatch.setattr(evaluation, "DegradedMNIST", lambda *args: object())
    monkeypatch.setattr(evaluation, "DataLoader", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="Evaluation produced no predictions"):
        evaluate_condition(
            model=SimpleCNN(),
            device=torch.device("cpu"),
            data_dir="data",
            batch_size=64,
            degradation="none",
            severity=0,
            seed=42,
            max_eval_batches=None
        )

def test_evaluate_condition_returns_prediction_rows(monkeypatch):
    # use one small batch so the prediction output can be checked directly
    loader = [(
        torch.zeros((2, 1, 28, 28)),
        torch.tensor([0, 1]),
        torch.tensor([10, 11])
    )]
    monkeypatch.setattr(evaluation, "DegradedMNIST", lambda *args: object())
    monkeypatch.setattr(evaluation, "DataLoader", lambda *args, **kwargs: loader)
    rows = evaluate_condition(
        model=SimpleCNN(),
        device=torch.device("cpu"),
        data_dir="data",
        batch_size=64,
        degradation="none",
        severity=0,
        seed=42,
        max_eval_batches=None
    )
    assert len(rows) == 2
    assert rows[0]["image_id"] == 10
    assert rows[0]["true_label"] == 0
    assert rows[0]["degradation"] == "none"
    assert 0 <= rows[0]["confidence"] <= 1

def test_summary_saves_evaluation_settings():
    # check that the summary keeps the main settings
    rows = [
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "seed": 42,
            "degradation": "none",
            "severity": 0,
            "correct": 1,
            "confidence": 0.95
        },
        {
            "dataset": "MNIST",
            "model": "SimpleCNN",
            "seed": 42,
            "degradation": "none",
            "severity": 0,
            "correct": 0,
            "confidence": 0.60
        }
    ]
    summary = summarise_condition(
        rows=rows,
        n_bins=5,
        fixed_hcer_threshold=0.90,
        adaptive_hcer_threshold=0.60,
        adaptive_hcer_percentile=90
    )

    assert summary["seed"] == 42
    assert summary["ece_bins"] == 5
    assert summary["fixed_hcer_threshold"] == 0.90
    assert summary["adaptive_hcer_threshold"] == 0.60
    assert summary["adaptive_hcer_percentile"] == 90
    assert summary["hcer"] == summary["hcer_fixed"]
    assert summary["hcer_fixed"] == 0.0
    assert summary["hcer_adaptive"] == 0.5
    assert summary["num_examples"] == 2