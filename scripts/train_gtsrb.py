"""Train and save the GTSRB baseline model."""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from src.datasets.gtsrb import GTSRB_CLASS_COUNT, GTSRB_IMAGE_SIZE, build_gtsrb_train_validation_datasets
from src.models.checkpoints import save_model_checkpoint
from src.models.gtsrb_cnn import GTSRBCNN
from src.utils.config import load_config, save_config_copy
from src.utils.seeds import set_seed

def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    max_train_batches: Optional[int]
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0

        for batch_index, (images, labels, _) in enumerate(loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        if batches == 0:
            raise ValueError("Training produced no batches.")

        average_loss = running_loss / batches
        print(f"Epoch {epoch}/{epochs} average loss: {average_loss:.4f}")

@torch.no_grad()
def calculate_validation_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    labels_all = []
    predictions_all = []

    for images, labels, _ in loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1).cpu()

        labels_all.extend(labels.tolist())
        predictions_all.extend(predictions.tolist())

    if not labels_all:
        raise ValueError("Validation loader produced no examples.")

    correct = sum(label == prediction for label, prediction in zip(labels_all, predictions_all))
    accuracy = correct / len(labels_all)
    balanced_accuracy = balanced_accuracy_score(labels_all, predictions_all)
    return accuracy, float(balanced_accuracy)

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GTSRB baseline CNN checkpoint")
    parser.add_argument("--config", default="configs/gtsrb.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "GTSRB" or config["model"] != "GTSRBCNN":
        raise ValueError("GTSRB training requires dataset GTSRB and model GTSRBCNN.")

    training_config = config["training"]
    set_seed(config["seed"])

    train_set, validation_set = build_gtsrb_train_validation_datasets(
        data_dir=config["data_dir"],
        validation_size=training_config["validation_size"],
        seed=config["seed"]
    )

    train_generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_set,
        batch_size=training_config["batch_size"],
        shuffle=True,
        generator=train_generator
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=training_config["batch_size"],
        shuffle=False
    )

    device = select_device()
    print(f"Using device: {device}")
    print(f"Training examples: {len(train_set)}")
    print(f"Validation examples: {len(validation_set)}")

    model = GTSRBCNN(num_classes=GTSRB_CLASS_COUNT).to(device)
    train_model(
        model=model,
        loader=train_loader,
        device=device,
        epochs=training_config["epochs"],
        learning_rate=training_config["learning_rate"],
        max_train_batches=training_config["max_train_batches"]
    )

    validation_accuracy, validation_balanced_accuracy = calculate_validation_metrics(
        model,
        validation_loader,
        device
    )
    checkpoint_path = Path(config["checkpoint"])
    config_copy_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_config.yaml")

    metadata = {
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "epochs": training_config["epochs"],
        "batch_size": training_config["batch_size"],
        "learning_rate": training_config["learning_rate"],
        "train_size": len(train_set),
        "validation_size": len(validation_set),
        "validation_accuracy": validation_accuracy,
        "validation_balanced_accuracy": validation_balanced_accuracy,
        "class_count": GTSRB_CLASS_COUNT,
        "image_size": GTSRB_IMAGE_SIZE,
        "training_augmentation": training_config["augmentation"],
        "device": str(device),
        "config": str(config_copy_path)
    }

    save_model_checkpoint(model, checkpoint_path, metadata)
    save_config_copy(config_path, config_copy_path)

    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Validation balanced accuracy: {validation_balanced_accuracy:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved config to {config_copy_path}")

if __name__ == "__main__":
    main()