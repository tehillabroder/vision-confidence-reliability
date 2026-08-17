"""Tests for shared degradation evaluation helpers."""

import pytest
import torch
import torch.nn as nn
from src.evaluation.runner import build_experiment_conditions, collect_prediction_rows

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