"""Minimal MNIST baseline prototype.

This trains a small CNN on MNIST, evaluates undegraded test images and saves
prediction-level results. It is a proof of concept for the later reliability
framework, not an attempt to train the best classifier.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.metrics.basic import (
    accuracy_from_correct,
    confidence_accuracy_gap,
    mean_confidence,
)
from src.models.simple_cnn import SimpleCNN

DATA_DIR = "data"


def set_seed(seed: int):
    # keep runs repeatable
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_loaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST mean and std
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, loader, device, epochs: int, max_train_batches=None):
    model.train()
    # standard baseline step size (0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0
        for images, labels in loader:
            # stop early when a batch cap is set
            if max_train_batches is not None and batches >= max_train_batches:
                break
            batches += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} avg loss {running_loss / max(batches, 1):.4f}")


@torch.no_grad()
def evaluate_with_predictions(model, loader, device):
    model.eval()

    rows =[]
    correct = 0
    total = 0
    image_id = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        # convert scores to probabilities summing to 1
        probabilities = F.softmax(logits, dim=1)
        # extract top class guess and its confidence level
        confidences, predictions = probabilities.max(dim=1)

        # compare vs true labels to check correctness
        batch_correct = predictions.eq(labels)
        correct += batch_correct.sum().item()
        total += labels.size(0)

        # log individual sample data for saving to csv
        for i in range(labels.size(0)):
            rows.append({
                "dataset": "MNIST",
                "model": "SimpleCNN",
                "image_id": image_id,
                "true_label": int(labels[i].item()),
                "predicted_label": int(predictions[i].item()),
                "correct": int(batch_correct[i].item()),
                "confidence": float(confidences[i].item()),
                "degradation": "none",
                "severity": 0,
            })
            image_id += 1

    accuracy = correct / total
    return accuracy, rows

def save_outputs(rows, accuracy: float, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(output_path / "predictions.csv", index=False)

    # convert pandas series to regular python lists for metric calculations
    correct = predictions_df["correct"].to_list()
    confidences = predictions_df["confidence"].to_list()

    metrics_df = pd.DataFrame([{
        "dataset": "MNIST",
        "model": "SimpleCNN",
        "degradation": "none",
        "severity": 0,
        "accuracy": accuracy_from_correct(correct),
        "mean_confidence": mean_confidence(confidences),
        "confidence_accuracy_gap": confidence_accuracy_gap(correct, confidences),
        "num_examples": len(predictions_df),
    }])

    metrics_df.to_csv(output_path / "metrics_summary.csv", index=False)

    print(f"Saved predictions to {output_path / 'predictions.csv'}")
    print(f"Saved metrics to {output_path / 'metrics_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Minimal MNIST baseline")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs (brief by default)")
    parser.add_argument("--batch-size", type=int, default=64)
    # enforce reproducibility across runs
    parser.add_argument("--seed", type=int, default=42, help="fixed random seed")
    # cap dataset size to speed up debugging/tests
    parser.add_argument("--max-train-batches", type=int, default=None, help="cap training batches per epoch for quick debugging")
    parser.add_argument("--output-dir", type=str, default="results/mnist_undegraded")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size)

    print(f"Train samples {len(train_loader.dataset)}")
    print(f"Test samples {len(test_loader.dataset)}")

    model = SimpleCNN().to(device)

    train(model, train_loader, device, args.epochs, args.max_train_batches)

    accuracy, rows = evaluate_with_predictions(model, test_loader, device)
    print(f"Undegraded test accuracy: {accuracy * 100:.2f}%")

    save_outputs(rows, accuracy, args.output_dir)

if __name__ == "__main__":
    main()
