"""Run MNIST evaluation across degradation severity levels."""

import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.mnist import DegradedMNIST
from src.evaluation.runner import build_calibration_rows, build_experiment_conditions, collect_prediction_rows, save_core_evaluation_outputs
from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.metrics.reliability import expected_calibration_error, high_confidence_error_rate
from src.models.checkpoints import load_model_checkpoint
from src.models.simple_cnn import SimpleCNN
from src.utils.seeds import set_seed
from src.utils.config import load_config
from src.evaluation.validation_profile import load_validation_profile, validate_validation_profile_source


def validate_evaluation_settings(ece_bins: int, hcer_threshold: float) -> None:
    if ece_bins <= 0:
        raise ValueError("ECE bin count must be greater than zero.")
    if not 0.0 <= hcer_threshold <= 1.0:
        raise ValueError("HCER threshold must be between 0 and 1.")
    
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

    return collect_prediction_rows(
        model=model,
        loader=loader,
        device=device,
        dataset_name="MNIST",
        model_name="SimpleCNN",
        degradation=degradation,
        severity=severity,
        seed=seed,
        max_eval_batches=max_eval_batches
    )

def summarise_condition(
    rows: list[dict],
    n_bins: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_threshold: float,
    adaptive_hcer_percentile: float
) -> dict:
    validate_evaluation_settings(n_bins, fixed_hcer_threshold)
    validate_evaluation_settings(n_bins, adaptive_hcer_threshold)

    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    first_row = rows[0]
    fixed_hcer = high_confidence_error_rate(correct, confidences, threshold=fixed_hcer_threshold)
    adaptive_hcer = high_confidence_error_rate(correct, confidences, threshold=adaptive_hcer_threshold)

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
        # keep the original column until the trust and plotting code is updated
        "hcer": fixed_hcer,
        "hcer_fixed": fixed_hcer,
        "hcer_adaptive": adaptive_hcer,
        "ece_bins": n_bins,
        "fixed_hcer_threshold": fixed_hcer_threshold,
        "adaptive_hcer_threshold": adaptive_hcer_threshold,
        "adaptive_hcer_percentile": adaptive_hcer_percentile,
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
    validation_profile = load_validation_profile(Path(config["validation_profile"]))
    validate_validation_profile_source(
        profile=validation_profile,
        dataset=config["dataset"],
        model=config["model"],
        checkpoint=config["checkpoint"],
        seed=config["seed"],
        fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
        adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"]
    )

    adaptive_hcer_threshold = validation_profile["adaptive_hcer_threshold"]
    validate_evaluation_settings(
        evaluation_config["ece_bins"],
        evaluation_config["fixed_hcer_threshold"]
    )
    validate_evaluation_settings(
        evaluation_config["ece_bins"],
        adaptive_hcer_threshold
    )
    set_seed(config["seed"])

    output_path = Path(config["output_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, metadata = load_evaluation_model(Path(config["checkpoint"]), device)
    if "validation_accuracy" in metadata:
        print(f"Checkpoint validation accuracy: {metadata['validation_accuracy']:.4f}")

    all_prediction_rows = []
    all_metric_rows = []
    all_calibration_rows = []

    experiment_conditions = build_experiment_conditions(evaluation_config["degradations"], evaluation_config["severity_levels"])

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
                rows=rows,
                n_bins=evaluation_config["ece_bins"],
                fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
                adaptive_hcer_threshold=adaptive_hcer_threshold,
                adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"]
            )
        )

        calibration_metadata = {
            "dataset": config["dataset"],
            "model": config["model"],
            "seed": config["seed"],
            # ece_bins remains before degradation and severity on purpose to match existing MNIST output schema
            "ece_bins": evaluation_config["ece_bins"],
            "degradation": degradation,
            "severity": severity
        }
        all_calibration_rows.extend(build_calibration_rows(rows, calibration_metadata))

    paths = save_core_evaluation_outputs(
        prediction_rows=all_prediction_rows,
        metric_rows=all_metric_rows,
        calibration_rows=all_calibration_rows,
        config_path=config_path,
        output_dir=output_path
    )
    print(f"Saved predictions to {paths['predictions']}")
    print(f"Saved metrics to {paths['metrics']}")
    print(f"Saved calibration bins to {paths['calibration']}")
    print(f"Saved config to {paths['config']}")

if __name__ == "__main__":
    main()