"""Build the MNIST undegraded validation reference profile."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.mnist import build_mnist_train_validation_datasets
from src.evaluation.validation_profile import (
    build_validation_profile,
    collect_validation_predictions,
    save_validation_profile
)
from src.models.checkpoints import load_model_checkpoint, validate_checkpoint_source
from src.models.simple_cnn import SimpleCNN
from src.utils.config import load_config
from src.utils.seeds import set_seed
from src.utils.outputs import check_output_paths

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MNIST validation reference profile")
    parser.add_argument("--config", default="configs/mnist.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Allow the existing validation profile to be replaced.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if config["dataset"] != "MNIST" or config["model"] != "SimpleCNN":
        raise ValueError("MNIST validation requires dataset MNIST and model SimpleCNN.")

    output_path = Path(config["validation_profile"])
    check_output_paths([output_path], overwrite=args.overwrite)

    training_config = config["training"]
    evaluation_config = config["evaluation"]
    set_seed(config["seed"])

    # use the training split helper so the test set is never loaded
    _, validation_set = build_mnist_train_validation_datasets(
        data_dir=config["data_dir"],
        validation_size=training_config["validation_size"],
        seed=config["seed"]
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=evaluation_config["batch_size"],
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Validation examples: {len(validation_set)}")

    checkpoint_path = Path(config["checkpoint"])

    model = SimpleCNN().to(device)
    metadata = load_model_checkpoint(model, checkpoint_path, device)
    # check the saved checkpoint before using it to define the validation baseline
    validate_checkpoint_source(
        metadata,
        {
            "dataset": config["dataset"],
            "model": config["model"],
            "seed": config["seed"]
        },
        "MNIST configuration"
    )

    correct, _, _, confidences = collect_validation_predictions(
        model,
        validation_loader,
        device
    )
    profile = build_validation_profile(
        dataset=config["dataset"],
        model=config["model"],
        checkpoint=str(checkpoint_path),
        seed=config["seed"],
        correct=correct,
        confidences=confidences,
        ece_bins=evaluation_config["ece_bins"],
        fixed_hcer_threshold=evaluation_config["fixed_hcer_threshold"],
        adaptive_hcer_percentile=evaluation_config["adaptive_hcer_percentile"]
    )
    output_path = save_validation_profile(profile, output_path)

    print(f"Baseline accuracy: {profile['baseline_accuracy']:.4f}")
    print(f"Baseline ECE: {profile['baseline_ece']:.4f}")
    print(f"Adaptive HCER threshold: {profile['adaptive_hcer_threshold']:.4f}")
    print(f"Saved validation profile to {output_path}")

if __name__ == "__main__":
    main()