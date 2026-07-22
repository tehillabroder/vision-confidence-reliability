"""Run GTSRB evaluation across degradation severity levels."""

import argparse
import json

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from src.datasets.gtsrb import GTSRB_CLASS_COUNT, build_gtsrb_test_dataset
from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.metrics.reliability import calibration_bins, expected_calibration_error, high_confidence_error_rate
from src.models.checkpoints import load_model_checkpoint
from src.models.gtsrb_cnn import GTSRBCNN
from src.utils.config import load_config, save_config_copy
from src.datasets.gtsrb_split import validate_gtsrb_split_metadata
from src.utils.seeds import set_seed

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_experiment_conditions(
    degradations: list[str],
    severity_levels: list[int]
) -> list[tuple[str, int]]:
    """Build the undegraded and degraded evaluation conditions."""
    conditions = [("none", 0)]
    for degradation in degradations:
        for severity in severity_levels:
            conditions.append((degradation, severity))
    return conditions

def load_validation_profile(profile_path: Path, config: dict) -> dict:
    """Load and validate the undegraded validation profile."""
    if not profile_path.exists():
        raise FileNotFoundError(f"Validation profile not found: {profile_path}")

    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid validation profile JSON: {profile_path}") from error

    if not isinstance(profile, dict):
        raise ValueError("Validation profile must contain a mapping.")

    expected_values = {
        "dataset": config["dataset"],
        "model": config["model"],
        "checkpoint": config["checkpoint"],
        "seed": config["seed"],
        "fixed_hcer_threshold": config["evaluation"]["fixed_hcer_threshold"],
        "adaptive_hcer_percentile": config["evaluation"]["adaptive_hcer_percentile"]
    }

    for name, expected_value in expected_values.items():
        if profile.get(name) != expected_value:
            raise ValueError(f"Validation profile {name} does not match the GTSRB configuration.")

    if profile.get("degradation") != "none" or profile.get("severity") != 0:
        raise ValueError("Validation profile must describe the undegraded condition.")

    training_config = config["training"]
    split_metadata = validate_gtsrb_split_metadata(
        metadata=profile.get("split_metadata"),
        validation_split=training_config["validation_split"],
        requested_validation_size=training_config["validation_size"],
        class_count=GTSRB_CLASS_COUNT
    )

    split_values = {
        "validation_split": split_metadata["validation_split"],
        "requested_validation_size": split_metadata["requested_validation_size"],
        "validation_track_count": split_metadata["validation_track_count"],
        "track_overlap": split_metadata["track_overlap"],
        "validation_track_hash": split_metadata["validation_track_hash"],
        "validation_sample_count": split_metadata["validation_size"]
    }

    for name, expected_value in split_values.items():
        if profile.get(name) != expected_value:
            raise ValueError(f"Validation profile {name} does not match its split metadata.")

    adaptive_threshold = profile.get("adaptive_hcer_threshold")
    if (
        isinstance(adaptive_threshold, bool)
        or not isinstance(adaptive_threshold, (int, float))
        or not 0 <= adaptive_threshold <= 1
    ):
        raise ValueError("Validation profile adaptive_hcer_threshold must be between 0 and 1.")

    return profile

def validate_evaluation_sources(
    metadata: dict,
    profile: dict,
    config: dict
) -> dict[str, object]:
    """Check that checkpoint and profile share one track split."""
    training_config = config["training"]
    checkpoint_split = validate_gtsrb_split_metadata(
        metadata=metadata.get("split_metadata"),
        validation_split=training_config["validation_split"],
        requested_validation_size=training_config["validation_size"],
        class_count=GTSRB_CLASS_COUNT
    )
    profile_split = validate_gtsrb_split_metadata(
        metadata=profile.get("split_metadata"),
        validation_split=training_config["validation_split"],
        requested_validation_size=training_config["validation_size"],
        class_count=GTSRB_CLASS_COUNT
    )

    expected_metadata = {
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "class_count": GTSRB_CLASS_COUNT,
        "validation_split": checkpoint_split["validation_split"],
        "requested_validation_size": checkpoint_split["requested_validation_size"],
        "validation_size": checkpoint_split["validation_size"],
        "track_overlap": 0,
        "validation_track_hash": checkpoint_split["validation_track_hash"]
    }

    for name, expected_value in expected_metadata.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"Checkpoint metadata {name} does not match "
                "the GTSRB evaluation configuration."
            )

    if checkpoint_split != profile_split:
        raise ValueError(
            "Checkpoint and validation profile use "
            "different GTSRB track splits."
        )

    return checkpoint_split

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
    """Evaluate one GTSRB degradation condition."""
    # reuse noise draws so severity comparisons remain controlled
    set_seed(seed)
    dataset = build_gtsrb_test_dataset(
        data_dir=data_dir,
        degradation=degradation,
        severity=severity
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    rows = []

    for batch_index, (images, labels, image_ids) in enumerate(loader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break

        images = images.to(device)
        labels = labels.to(device)
        probabilities = torch.softmax(model(images), dim=1)
        confidences, predictions = probabilities.max(dim=1)
        batch_correct = predictions.eq(labels)

        for index in range(labels.size(0)):
            rows.append({
                "dataset": "GTSRB",
                "model": "GTSRBCNN",
                "seed": seed,
                "image_id": int(image_ids[index].item()),
                "true_label": int(labels[index].item()),
                "predicted_label": int(predictions[index].item()),
                "correct": int(batch_correct[index].item()),
                "confidence": float(confidences[index].item()),
                "degradation": degradation,
                "severity": severity
            })

    if not rows:
        raise ValueError("Evaluation produced no predictions.")

    return rows

def summarise_condition(
    rows: list[dict],
    ece_bins: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_threshold: float,
    adaptive_hcer_percentile: float,
    split_metadata: dict[str, object]
) -> dict:
    """Summarise one GTSRB evaluation condition."""
    if not rows:
        raise ValueError("Cannot summarise an empty evaluation condition.")

    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]
    first_row = rows[0]

    hcer_fixed = high_confidence_error_rate(
        correct,
        confidences,
        threshold=fixed_hcer_threshold
    )
    hcer_adaptive = high_confidence_error_rate(
        correct,
        confidences,
        threshold=adaptive_hcer_threshold
    )

    return {
        "dataset": first_row["dataset"],
        "model": first_row["model"],
        "seed": first_row["seed"],
        "degradation": first_row["degradation"],
        "severity": first_row["severity"],
        "validation_split": split_metadata["validation_split"],
        "requested_validation_size": split_metadata["requested_validation_size"],
        "validation_size": split_metadata["validation_size"],
        "validation_track_count": split_metadata["validation_track_count"],
        "track_overlap": split_metadata["track_overlap"],
        "validation_track_hash": split_metadata["validation_track_hash"],
        "accuracy": accuracy_from_correct(correct),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "mean_confidence": mean_confidence(confidences),
        "confidence_accuracy_gap": confidence_accuracy_gap(correct, confidences),
        "ece": expected_calibration_error(correct, confidences, n_bins=ece_bins),
        "hcer": hcer_fixed,
        "hcer_fixed": hcer_fixed,
        "hcer_adaptive": hcer_adaptive,
        "fixed_hcer_threshold": fixed_hcer_threshold,
        "adaptive_hcer_threshold": adaptive_hcer_threshold,
        "adaptive_hcer_percentile": adaptive_hcer_percentile,
        "ece_bins": ece_bins,
        "num_examples": len(rows)
    }

def build_calibration_rows(rows: list[dict], ece_bins: int) -> list[dict]:
    """Build calibration-bin rows with condition metadata."""
    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    first_row = rows[0]
    condition_rows = calibration_bins(correct, confidences, n_bins=ece_bins)

    for row in condition_rows:
        row["dataset"] = first_row["dataset"]
        row["model"] = first_row["model"]
        row["seed"] = first_row["seed"]
        row["degradation"] = first_row["degradation"]
        row["severity"] = first_row["severity"]
        row["ece_bins"] = ece_bins

    return condition_rows

def save_evaluation_outputs(
    prediction_rows: list[dict],
    metric_rows: list[dict],
    calibration_rows: list[dict],
    split_metadata: dict[str, object],
    config_path: Path,
    output_dir: Path
) -> dict[str, Path]:
    """Save GTSRB evaluation evidence."""
    if not prediction_rows or not metric_rows or not calibration_rows:
        raise ValueError("Evaluation outputs must not be empty.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": output_dir / "predictions.csv",
        "metrics": output_dir / "metrics_summary.csv",
        "calibration": output_dir / "calibration_bins.csv",
        "split_metadata": output_dir / "split_metadata.json",
        "config": output_dir / "config.yaml"
    }

    pd.DataFrame(prediction_rows).to_csv(paths["predictions"], index=False)
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    pd.DataFrame(calibration_rows).to_csv(paths["calibration"], index=False)
    paths["split_metadata"].write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8"
    )
    save_config_copy(config_path, paths["config"])
    return paths

def load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device
) -> tuple[nn.Module, dict]:
    """Load the saved GTSRB baseline model."""
    model = GTSRBCNN(num_classes=GTSRB_CLASS_COUNT).to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, metadata

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GTSRB degradation evaluation")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "GTSRB" or config["model"] != "GTSRBCNN":
        raise ValueError("GTSRB evaluation requires dataset GTSRB and model GTSRBCNN.")

    evaluation_config = config["evaluation"]
    profile = load_validation_profile(
        Path(config["validation_profile"]),
        config
    )

    device = select_device()
    model, metadata = load_evaluation_model(
        Path(config["checkpoint"]),
        device
    )
    split_metadata = validate_evaluation_sources(metadata, profile, config)

    print(f"Using device: {device}")
    if "validation_accuracy" in metadata:
        print(
            "Checkpoint validation accuracy: "
            f"{metadata['validation_accuracy']:.4f}"
        )
    if "validation_balanced_accuracy" in metadata:
        print(
            "Checkpoint validation balanced accuracy: "
            f"{metadata['validation_balanced_accuracy']:.4f}"
        )

    prediction_rows = []
    metric_rows = []
    calibration_rows = []
    conditions = build_experiment_conditions(
        evaluation_config["degradations"],
        evaluation_config["severity_levels"]
    )

    for degradation, severity in conditions:
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
        prediction_rows.extend(rows)
        metric_rows.append(
            summarise_condition(
                rows=rows,
                ece_bins=evaluation_config["ece_bins"],
                fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
                adaptive_hcer_threshold=profile["adaptive_hcer_threshold"],
                adaptive_hcer_percentile=profile["adaptive_hcer_percentile"],
                split_metadata=split_metadata
            )
        )
        calibration_rows.extend(
            build_calibration_rows(
                rows,
                evaluation_config["ece_bins"]
            )
        )

    paths = save_evaluation_outputs(
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        calibration_rows=calibration_rows,
        split_metadata=split_metadata,
        config_path=config_path,
        output_dir=Path(config["output_dir"])
    )

    print(f"Saved predictions to {paths['predictions']}")
    print(f"Saved metrics to {paths['metrics']}")
    print(f"Saved calibration bins to {paths['calibration']}")
    print(f"Saved split metadata to {paths['split_metadata']}")
    print(f"Saved config to {paths['config']}")

if __name__ == "__main__":
    main()