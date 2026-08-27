"""Run GTSRB evaluation across degradation severity levels."""

import argparse
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from src.datasets.gtsrb import GTSRB_CLASS_COUNT, build_gtsrb_test_dataset
from src.evaluation.runner import (
    build_calibration_rows, build_core_evaluation_output_paths, build_experiment_conditions,
    collect_prediction_rows, save_core_evaluation_outputs, validate_evaluation_settings
)
from src.evaluation.validation_profile import load_validation_profile, validate_validation_profile_source
from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.metrics.reliability import (
    expected_calibration_error, high_confidence_coverage, high_confidence_error_rate, 
    rank_based_high_confidence_coverage, rank_based_high_confidence_error_rate
)
from src.models.checkpoints import load_model_checkpoint
from src.models.gtsrb_models import build_gtsrb_model
from src.utils.config import load_config
from src.utils.outputs import check_output_paths
from src.datasets.gtsrb_split import validate_gtsrb_split_metadata
from src.utils.seeds import set_seed

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def validate_gtsrb_validation_profile(profile: dict, config: dict) -> dict[str, object]:
    """Validate GTSRB-specific validation profile evidence."""
    evaluation_config = config["evaluation"]

    validate_validation_profile_source(
        profile=profile,
        dataset=config["dataset"],
        model=config["model"],
        checkpoint=config["checkpoint"],
        seed=config["seed"],
        fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
        adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"]
    )

    if profile.get("rank_hcer_top_fraction") != evaluation_config["rank_hcer_top_fraction"]:
        raise ValueError("Validation profile rank_hcer_top_fraction does not match the GTSRB configuration.")

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

    return split_metadata

def validate_evaluation_sources(
    metadata: dict,
    profile_split: dict[str, object],
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

def evaluate_condition(
    model: nn.Module,
    device: torch.device,
    model_name: str,
    data_dir: str,
    batch_size: int,
    degradation: str,
    severity: int,
    seed: int,
    max_eval_batches: Optional[int]
) -> list[dict]:
    """Evaluate one GTSRB degradation condition."""
    # reuse identical noise pattern so severity comparisons remain controlled
    set_seed(seed)
    dataset = build_gtsrb_test_dataset(data_dir=data_dir, degradation=degradation, severity=severity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return collect_prediction_rows(
        model=model,
        loader=loader,
        device=device,
        dataset_name="GTSRB",
        model_name=model_name,
        degradation=degradation,
        severity=severity,
        seed=seed,
        max_eval_batches=max_eval_batches
    )

def summarise_condition(
    rows: list[dict],
    ece_bins: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_threshold: float,
    adaptive_hcer_percentile: float,
    rank_hcer_top_fraction: float,
    split_metadata: dict[str, object]
) -> dict:
    """Summarise one GTSRB evaluation condition."""
    if not rows:
        raise ValueError("Cannot summarise an empty evaluation condition.")

    validate_evaluation_settings(ece_bins, fixed_hcer_threshold, adaptive_hcer_threshold)

    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    true_labels = [row["true_label"] for row in rows]
    predicted_labels = [row["predicted_label"] for row in rows]
    first_row = rows[0]

    hcer_fixed = high_confidence_error_rate(correct, confidences, threshold=fixed_hcer_threshold)
    hcer_adaptive = high_confidence_error_rate(correct, confidences, threshold=adaptive_hcer_threshold)
    hcer_adaptive_coverage = high_confidence_coverage(confidences, threshold=adaptive_hcer_threshold)
    hcer_rank = rank_based_high_confidence_error_rate(correct, confidences, top_fraction=rank_hcer_top_fraction)
    hcer_rank_coverage = rank_based_high_confidence_coverage(confidences, top_fraction=rank_hcer_top_fraction)

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
        "hcer_adaptive_coverage": hcer_adaptive_coverage,
        "hcer_rank": hcer_rank,
        "hcer_rank_coverage": hcer_rank_coverage,
        "rank_hcer_top_fraction": rank_hcer_top_fraction,
        "fixed_hcer_threshold": fixed_hcer_threshold,
        "adaptive_hcer_threshold": adaptive_hcer_threshold,
        "adaptive_hcer_percentile": adaptive_hcer_percentile,
        "ece_bins": ece_bins,
        "num_examples": len(rows)
    }

def save_evaluation_outputs(
    prediction_rows: list[dict],
    metric_rows: list[dict],
    calibration_rows: list[dict],
    split_metadata: dict[str, object],
    config_path: Path,
    output_dir: Path
) -> dict[str, Path]:
    """Save GTSRB evaluation evidence."""

    core_paths = save_core_evaluation_outputs(
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        calibration_rows=calibration_rows,
        config_path=config_path,
        output_dir=output_dir
    )
    split_metadata_path = output_dir / "split_metadata.json"
    split_metadata_path.write_text(json.dumps(split_metadata, indent=2), encoding="utf-8")

    return {
        "predictions": core_paths["predictions"],
        "metrics": core_paths["metrics"],
        "calibration": core_paths["calibration"],
        "split_metadata": split_metadata_path,
        "config": core_paths["config"]
    }

def load_evaluation_model(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str
) -> tuple[nn.Module, dict]:
    """Load one saved GTSRB model for evaluation."""
    
    model = build_gtsrb_model(model_name=model_name, num_classes=GTSRB_CLASS_COUNT, pretrained_weights=None).to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, metadata

def main() -> None:
    parser = argparse.ArgumentParser(description="Run GTSRB degradation evaluation")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing evaluation evidence to be replaced.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "GTSRB":
        raise ValueError("GTSRB evaluation requires dataset GTSRB.")

    output_dir = Path(config["output_dir"])
    output_paths = build_core_evaluation_output_paths(output_dir)
    check_output_paths([*output_paths.values(), output_dir / "split_metadata.json"], overwrite=args.overwrite)

    evaluation_config = config["evaluation"]

    profile = load_validation_profile(Path(config["validation_profile"]))
    profile_split = validate_gtsrb_validation_profile(profile, config)

    validate_evaluation_settings(evaluation_config["ece_bins"], evaluation_config["fixed_hcer_threshold"], profile["adaptive_hcer_threshold"])

    device = select_device()
    model, metadata = load_evaluation_model(
        Path(config["checkpoint"]),
        device, 
        config["model"]
    )
    split_metadata = validate_evaluation_sources(metadata, profile_split, config)

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
            model_name=config["model"],
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
                rank_hcer_top_fraction=profile["rank_hcer_top_fraction"],
                split_metadata=split_metadata
            )
        )
        calibration_rows.extend(build_calibration_rows(rows, evaluation_config["ece_bins"]))

    paths = save_evaluation_outputs(
        prediction_rows=prediction_rows,
        metric_rows=metric_rows,
        calibration_rows=calibration_rows,
        split_metadata=split_metadata,
        config_path=config_path,
        output_dir=output_dir
    )

    print(f"Saved predictions to {paths['predictions']}")
    print(f"Saved metrics to {paths['metrics']}")
    print(f"Saved calibration bins to {paths['calibration']}")
    print(f"Saved split metadata to {paths['split_metadata']}")
    print(f"Saved config to {paths['config']}")

if __name__ == "__main__":
    main()