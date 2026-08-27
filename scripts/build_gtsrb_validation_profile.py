"""Build the GTSRB undegraded validation profile."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from scripts.train_gtsrb import select_device
from src.datasets.gtsrb import GTSRB_CLASS_COUNT, build_gtsrb_train_validation_split
from src.datasets.gtsrb_split import validate_gtsrb_split_metadata
from src.evaluation.validation_profile import build_validation_profile, collect_validation_predictions, save_validation_profile
from src.metrics.reliability import high_confidence_coverage, high_confidence_error_rate, rank_based_high_confidence_coverage, rank_based_high_confidence_error_rate
from src.models.checkpoints import load_model_checkpoint
from src.models.gtsrb_models import build_gtsrb_model
from src.utils.config import load_config
from src.utils.seeds import set_seed
from src.utils.outputs import check_output_paths


def load_validation_model(checkpoint_path: Path, device: torch.device, model_name: str) -> tuple[nn.Module, dict]:
    """Load one saved GTSRB model for validation."""
    # the fine-tuned checkpoint already contains every parameter, so don't download ImageNet weights again
    model = build_gtsrb_model(model_name=model_name, num_classes=GTSRB_CLASS_COUNT, pretrained_weights=None).to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model, metadata

def build_gtsrb_validation_profile(
    dataset: str,
    model_name: str,
    checkpoint: str,
    seed: int,
    correct: list[int],
    true_labels: list[int],
    predicted_labels: list[int],
    confidences: list[float],
    ece_bins: int,
    fixed_hcer_threshold: float,
    adaptive_hcer_percentile: float,
    rank_hcer_top_fraction: float,
    split_metadata: dict[str, object]
) -> dict:
    """Build a GTSRB profile with track-split evidence."""
    if len(correct) != split_metadata["validation_size"]:
        raise ValueError(
            "Validation outputs do not match the saved split size."
        )

    profile = build_validation_profile(
        dataset=dataset,
        model=model_name,
        checkpoint=checkpoint,
        seed=seed,
        correct=correct,
        confidences=confidences,
        ece_bins=ece_bins,
        fixed_hcer_threshold=fixed_hcer_threshold,
        adaptive_hcer_percentile=adaptive_hcer_percentile
    )
    adaptive_threshold = profile["adaptive_hcer_threshold"]
    profile.update({
        "baseline_adaptive_hcer": high_confidence_error_rate(correct, confidences, threshold=adaptive_threshold),
        "baseline_adaptive_hcer_coverage": high_confidence_coverage(confidences, threshold=adaptive_threshold),
        "rank_hcer_top_fraction": rank_hcer_top_fraction,
        "baseline_rank_hcer": rank_based_high_confidence_error_rate(correct, confidences, top_fraction=rank_hcer_top_fraction),
        "baseline_rank_hcer_coverage": rank_based_high_confidence_coverage(confidences, top_fraction=rank_hcer_top_fraction),
        "baseline_balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels
            )
        ),
        "validation_split": split_metadata["validation_split"],
        "requested_validation_size": split_metadata[
            "requested_validation_size"
        ],
        "validation_track_count": split_metadata[
            "validation_track_count"
        ],
        "track_overlap": split_metadata["track_overlap"],
        "validation_track_hash": split_metadata[
            "validation_track_hash"
        ],
        "split_metadata": split_metadata
    })
    return profile

def validate_checkpoint_metadata(metadata: dict, config: dict, split_metadata: dict) -> None:
    """Check that the checkpoint matches the GTSRB configuration."""
    training_config = config["training"]
    split_metadata = validate_gtsrb_split_metadata(
        metadata=split_metadata,
        validation_split=training_config["validation_split"],
        requested_validation_size=training_config["validation_size"],
        class_count=GTSRB_CLASS_COUNT
    )
    checkpoint_split = validate_gtsrb_split_metadata(
        metadata=metadata.get("split_metadata"),
        validation_split=training_config["validation_split"],
        requested_validation_size=training_config["validation_size"],
        class_count=GTSRB_CLASS_COUNT
    )

    expected_values = {
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "class_count": GTSRB_CLASS_COUNT,
        "validation_split": split_metadata["validation_split"],
        "requested_validation_size": split_metadata[
            "requested_validation_size"
        ],
        "train_size": split_metadata["train_size"],
        "validation_size": split_metadata["validation_size"],
        "track_overlap": 0,
        "validation_track_hash": split_metadata[
            "validation_track_hash"
        ]
    }

    for name, expected_value in expected_values.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"Checkpoint metadata {name} does not match "
                "the GTSRB configuration."
            )

    if checkpoint_split != split_metadata:
        raise ValueError(
            "Checkpoint split metadata does not match "
            "the reconstructed GTSRB split."
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GTSRB validation profile")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Allow the existing validation profile to be replaced.")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "GTSRB":
        raise ValueError("GTSRB validation requires dataset GTSRB.")

    output_path = Path(config["validation_profile"])
    check_output_paths([output_path], overwrite=args.overwrite)

    training_config = config["training"]
    evaluation_config = config["evaluation"]
    set_seed(config["seed"])

    _, validation_set, split_metadata = (
        build_gtsrb_train_validation_split(
            data_dir=config["data_dir"],
            validation_size=training_config["validation_size"],
            seed=config["seed"],
            validation_split=training_config["validation_split"]
        )
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=evaluation_config["batch_size"],
        shuffle=False
    )

    device = select_device()
    checkpoint_path = Path(config["checkpoint"])
    model, metadata = load_validation_model(checkpoint_path, device, config["model"])
    # reject mismatched checkpoints before producing an invalid reference profile
    validate_checkpoint_metadata(metadata, config, split_metadata)

    correct, true_labels, predicted_labels, confidences = collect_validation_predictions(
        model,
        validation_loader,
        device
    )
    profile = build_gtsrb_validation_profile(
        dataset=config["dataset"],
        model_name=config["model"],
        checkpoint=str(checkpoint_path),
        seed=config["seed"],
        correct=correct,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        confidences=confidences,
        ece_bins=evaluation_config["ece_bins"],
        fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
        adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"],
        rank_hcer_top_fraction=evaluation_config["rank_hcer_top_fraction"],
        split_metadata=split_metadata
    )
    output_path = save_validation_profile(profile, output_path)
    
    print(f"Using device: {device}")
    print(f"Validation examples: {profile['validation_sample_count']}")
    print(f"Baseline accuracy: {profile['baseline_accuracy']:.4f}")
    # report balanced accuracy because GTSRB has unequal class frequencies
    print(
        "Baseline balanced accuracy: "
        f"{profile['baseline_balanced_accuracy']:.4f}"
    )
    print(f"Baseline ECE: {profile['baseline_ece']:.4f}")
    print(
        "Adaptive HCER threshold: "
        f"{profile['adaptive_hcer_threshold']:.4f}"
    )
    print(f"Adaptive HCER: {profile['baseline_adaptive_hcer']:.4f}")
    print(f"Adaptive HCER coverage: {profile['baseline_adaptive_hcer_coverage']:.4f}")
    print(f"Rank-based HCER: {profile['baseline_rank_hcer']:.4f}")
    print(f"Rank-based HCER coverage: {profile['baseline_rank_hcer_coverage']:.4f}")
    print(f"Track overlap: {profile['track_overlap']}")
    print(
        "Validation track hash: "
        f"{profile['validation_track_hash']}"
    )
    print(f"Saved validation profile to {output_path}")

if __name__ == "__main__":
    main()