"""Train and save the MNIST baseline model."""

import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.mnist import build_mnist_train_validation_datasets
from src.models.checkpoints import save_model_checkpoint
from src.models.simple_cnn import SimpleCNN
from src.utils.seeds import set_seed
from src.utils.config import load_config, save_config_copy

def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    max_train_batches: Optional[int]
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0

        for batch_index, (images, labels) in enumerate(loader):
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
def calculate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images).argmax(dim=1)

        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)

    if total == 0:
        raise ValueError("Validation loader produced no examples.")

    return correct / total

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST SimpleCNN checkpoint")
    parser.add_argument("--config", default="configs/mnist.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    if config["dataset"] != "MNIST" or config["model"] != "SimpleCNN":
        raise ValueError("MNIST training requires dataset MNIST and model SimpleCNN.")

    training_config = config["training"]
    set_seed(config["seed"])

    train_set, validation_set = build_mnist_train_validation_datasets(
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training examples: {len(train_set)}")
    print(f"Validation examples: {len(validation_set)}")

    model = SimpleCNN().to(device)
    train_model(
        model=model,
        loader=train_loader,
        device=device,
        epochs=training_config["epochs"],
        max_train_batches=training_config["max_train_batches"]
    )

    validation_accuracy = calculate_accuracy(model, validation_loader, device)
    checkpoint_path = Path(config["checkpoint"])
    config_copy_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_config.yaml")

    metadata = {
        "dataset": config["dataset"],
        "model": config["model"],
        "seed": config["seed"],
        "epochs": training_config["epochs"],
        "batch_size": training_config["batch_size"],
        "train_size": len(train_set),
        "validation_size": len(validation_set),
        "validation_accuracy": validation_accuracy,
        "training_augmentation": training_config["augmentation"],
        "config": str(config_copy_path)
    }

    save_model_checkpoint(model, checkpoint_path, metadata)
    save_config_copy(config_path, config_copy_path)

    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved config to {config_copy_path}")

if __name__ == "__main__":
    main()