"""Run MNIST evaluation across degradation severity levels."""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.mnist import DegradedMNIST
from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.metrics.reliability import calibration_bins, expected_calibration_error, high_confidence_error_rate
from src.models.checkpoints import load_model_checkpoint
from src.models.simple_cnn import SimpleCNN
from src.utils.seeds import set_seed
from src.utils.config import load_config, save_config_copy


def validate_evaluation_settings(ece_bins: int, hcer_threshold: float) -> None:
    if ece_bins <= 0:
        raise ValueError("ECE bin count must be greater than zero.")
    if not 0.0 <= hcer_threshold <= 1.0:
        raise ValueError("HCER threshold must be between 0 and 1.")
    
@torch.no_grad()
def evaluate_condition(
    model: nn.Module,
    device: torch.device,
    data_dir: str,
    batch_size: int,
    degradation: str,
    severity: int,
    seed: int,
    max_eval_batches: Optional[int]
) -> list[dict]:
    # reuse random draws so noise comparisons stay controlled
    set_seed(seed)
    dataset = DegradedMNIST(data_dir, degradation, severity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    rows = []
    
    for batch_index, (images, labels, image_ids) in enumerate(loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        
        images, labels = images.to(device), labels.to(device)
        probabilities = F.softmax(model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        batch_correct = predictions.eq(labels)
        for i in range(labels.size(0)):
            rows.append({
                "dataset": "MNIST",
                "model": "SimpleCNN",
                "seed": seed,
                "image_id": int(image_ids[i].item()),
                "true_label": int(labels[i].item()),
                "predicted_label": int(predictions[i].item()),
                "correct": int(batch_correct[i].item()),
                "confidence": float(confidences[i].item()),
                "degradation": degradation,
                "severity": severity
            })
    if not rows:
        raise ValueError("Evaluation produced no predictions.")
            
    return rows

def summarise_condition(rows: list[dict], n_bins: int, hcer_threshold: float) -> dict:
    validate_evaluation_settings(n_bins, hcer_threshold)
    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    first_row = rows[0]
    return {
        "dataset": first_row["dataset"],
        "model": first_row["model"],
        "seed": first_row["seed"],
        "degradation": first_row["degradation"],
        "severity": first_row["severity"],
        "accuracy": accuracy_from_correct(correct),
        "mean_confidence": mean_confidence(confidences),
        "confidence_accuracy_gap": confidence_accuracy_gap(correct, confidences),
        "ece": expected_calibration_error(correct, confidences, n_bins=n_bins),
        "hcer": high_confidence_error_rate(correct, confidences, threshold=hcer_threshold),
        "ece_bins": n_bins,
        "hcer_threshold": hcer_threshold,
        "num_examples": len(rows)
    }

def load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device
) -> tuple[nn.Module, dict]:
    model = SimpleCNN().to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, metadata

def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST degradation evaluation")
    parser.add_argument("--config", default="configs/mnist.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "MNIST" or config["model"] != "SimpleCNN":
        raise ValueError("MNIST evaluation requires dataset MNIST and model SimpleCNN.")

    evaluation_config = config["evaluation"]
    validate_evaluation_settings(
        evaluation_config["ece_bins"],
        evaluation_config["fixed_hcer_threshold"]
    )
    set_seed(config["seed"])

    output_path = Path(config["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, metadata = load_evaluation_model(Path(config["checkpoint"]), device)
    if "validation_accuracy" in metadata:
        print(f"Checkpoint validation accuracy: {metadata['validation_accuracy']:.4f}")

    all_prediction_rows = []
    all_metric_rows = []
    all_calibration_rows = []
    experiment_conditions = [("none", 0)]

    for degradation in evaluation_config["degradations"]:
        for severity in evaluation_config["severity_levels"]:
            experiment_conditions.append((degradation, severity))

    for degradation, severity in experiment_conditions:
        print(f"Evaluating {degradation}, severity {severity}")

        rows = evaluate_condition(
            model=model,
            device=device,
            data_dir=config["data_dir"],
            batch_size=evaluation_config["batch_size"],
            degradation=degradation,
            severity=severity,
            seed=config["seed"],
            max_eval_batches=evaluation_config["max_eval_batches"]
        )
        all_prediction_rows.extend(rows)
        all_metric_rows.append(
            summarise_condition(
                rows,
                evaluation_config["ece_bins"],
                evaluation_config["fixed_hcer_threshold"]
            )
        )

        correct = [row["correct"] for row in rows]
        confidences = [row["confidence"] for row in rows]
        condition_bins = calibration_bins(
            correct,
            confidences,
            n_bins=evaluation_config["ece_bins"]
        )

        for bin_row in condition_bins:
            bin_row["dataset"] = config["dataset"]
            bin_row["model"] = config["model"]
            bin_row["seed"] = config["seed"]
            bin_row["ece_bins"] = evaluation_config["ece_bins"]
            bin_row["degradation"] = degradation
            bin_row["severity"] = severity
            all_calibration_rows.append(bin_row)

    predictions_path = output_path / "predictions.csv"
    metrics_path = output_path / "metrics_summary.csv"
    calibration_path = output_path / "calibration_bins.csv"
    config_copy_path = output_path / "config.yaml"

    pd.DataFrame(all_prediction_rows).to_csv(predictions_path, index=False)
    pd.DataFrame(all_metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(all_calibration_rows).to_csv(calibration_path, index=False)
    save_config_copy(config_path, config_copy_path)

    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved calibration bins to {calibration_path}")
    print(f"Saved config to {config_copy_path}")

if __name__ == "__main__":
    main()