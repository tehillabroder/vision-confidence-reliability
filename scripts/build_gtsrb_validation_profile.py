"""Build the GTSRB undegraded validation profile."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from scripts.train_gtsrb import select_device
from src.datasets.gtsrb import GTSRB_CLASS_COUNT, build_gtsrb_train_validation_datasets
from src.evaluation.validation_profile import build_validation_profile, save_validation_profile
from src.models.checkpoints import load_model_checkpoint
from src.models.gtsrb_cnn import GTSRBCNN
from src.utils.config import load_config
from src.utils.seeds import set_seed

@torch.no_grad()
def collect_gtsrb_validation_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple[list[int], list[int], list[int], list[float]]:
    """Collect GTSRB validation predictions and confidence values."""
    model.eval()
    correct = []
    true_labels = []
    predicted_labels = []
    confidences = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        probabilities = torch.softmax(model(images), dim=1)
        batch_confidences, predictions = probabilities.max(dim=1)

        correct.extend(predictions.eq(labels).int().cpu().tolist())
        true_labels.extend(labels.cpu().tolist())
        predicted_labels.extend(predictions.cpu().tolist())
        confidences.extend(batch_confidences.cpu().tolist())

    if not true_labels:
        raise ValueError("Validation loader produced no examples.")

    return correct, true_labels, predicted_labels, confidences

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
    adaptive_hcer_percentile: float
) -> dict:
    """Build a GTSRB profile with balanced accuracy."""
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
    profile["baseline_balanced_accuracy"] = float(
        balanced_accuracy_score(true_labels, predicted_labels)
    )
    return profile

def validate_checkpoint_metadata(metadata: dict, config: dict) -> None:
    """Check that the checkpoint matches the GTSRB configuration."""
    expected_values = {
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "validation_size": config["training"]["validation_size"],
        "class_count": GTSRB_CLASS_COUNT
    }

    for name, expected_value in expected_values.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"Checkpoint metadata {name} does not match the GTSRB configuration."
            )

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GTSRB validation profile")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "GTSRB" or config["model"] != "GTSRBCNN":
        raise ValueError("GTSRB validation requires dataset GTSRB and model GTSRBCNN.")

    training_config = config["training"]
    evaluation_config = config["evaluation"]
    set_seed(config["seed"])

    # use the undegraded validation split as the reference for later deterioration
    _, validation_set = build_gtsrb_train_validation_datasets(
        data_dir=config["data_dir"],
        validation_size=training_config["validation_size"],
        seed=config["seed"]
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=evaluation_config["batch_size"],
        shuffle=False
    )

    device = select_device()
    checkpoint_path = Path(config["checkpoint"])
    model = GTSRBCNN(num_classes=GTSRB_CLASS_COUNT).to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    # reject mismatched checkpoints before producing an invalid reference profile
    validate_checkpoint_metadata(metadata, config)

    correct, true_labels, predicted_labels, confidences = collect_gtsrb_validation_outputs(
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
        adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"]
    )

    output_path = save_validation_profile(
        profile,
        Path(config["validation_profile"])
    )

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
    print(f"Saved validation profile to {output_path}")

if __name__ == "__main__":
    main()