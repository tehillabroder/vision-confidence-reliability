"""Run MNIST evaluation across degradation severity levels."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.degradations.image_degradations import apply_degradation

from src.metrics.basic import accuracy_from_correct, confidence_accuracy_gap, mean_confidence
from src.models.simple_cnn import SimpleCNN
from src.metrics.reliability import (calibration_bins, expected_calibration_error, high_confidence_error_rate,)

DATA_DIR = "data"
# standard mnist mean and standard deviation
NORMALISE = transforms.Normalize((0.1307,), (0.3081,))

def set_seed(seed: int):
    # keep runs repeatable across environments
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DegradedMNIST(Dataset):
    """MNIST test set with degradation applied before normalisation."""
    def __init__(self, degradation: str, severity: int):
        self.base_dataset = datasets.MNIST(
            DATA_DIR,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.degradation = degradation
        self.severity = severity
    def __len__(self):
        return len(self.base_dataset)
    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        degraded_image = apply_degradation(
            image,
            self.degradation,
            self.severity
        )
        normalised_image = NORMALISE(degraded_image)
        return normalised_image, label, index

def get_train_loader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        NORMALISE,
    ])
    train_set = datasets.MNIST(
        DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=True)

def train(model, loader, device, epochs: int, max_train_batches=None):
    model.train()
    # standard learning rate for adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0
        for images, labels in loader:
            if max_train_batches is not None and batches >= max_train_batches:
                break
            batches += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        average_loss = running_loss / max(batches, 1)  # avoid division by zero
        print(f"Epoch {epoch}/{epochs} average loss: {average_loss:.4f}")

@torch.no_grad()
def evaluate_condition(model, device, batch_size: int, degradation: str, severity: int, max_eval_batches=None):
    dataset = DegradedMNIST(degradation=degradation, severity=severity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    batches = 0
    model.eval()
    
    for images, labels, image_ids in loader:
        if max_eval_batches is not None and batches >= max_eval_batches:
            break
        batches += 1
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
        batch_correct = predictions.eq(labels)
        for i in range(labels.size(0)):
            rows.append({
                "dataset": "MNIST",
                "model": "SimpleCNN",
                "image_id": int(image_ids[i].item()),
                "true_label": int(labels[i].item()),
                "predicted_label": int(predictions[i].item()),
                "correct": int(batch_correct[i].item()),
                "confidence": float(confidences[i].item()),
                "degradation": degradation,
                "severity": severity
            })
    return rows

def summarise_condition(rows):
    correct = [row["correct"] for row in rows]
    confidences = [row["confidence"] for row in rows]
    first_row = rows[0]
    return {
        "dataset": first_row["dataset"],
        "model": first_row["model"],
        "degradation": first_row["degradation"],
        "severity": first_row["severity"],
        "accuracy": accuracy_from_correct(correct),
        "mean_confidence": mean_confidence(confidences),
        "confidence_accuracy_gap": confidence_accuracy_gap(correct, confidences),
        "ece": expected_calibration_error(correct, confidences, n_bins=10),
        # 0.90 is the initial high-confidence threshold
        "hcer": high_confidence_error_rate(correct, confidences, threshold=0.90),
        "num_examples": len(rows)
    }

def main():
    parser = argparse.ArgumentParser(description="MNIST degradation evaluation")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/mnist_degradation_eval")

    args = parser.parse_args()
    set_seed(args.seed)

    # ensure output directory exists before saving results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = get_train_loader(args.batch_size)

    model = SimpleCNN().to(device)
    
    train(model, train_loader, device, args.epochs, args.max_train_batches)
    
    all_prediction_rows = []
    all_metric_rows = []
    experiment_conditions = [("none", 0)]
    all_calibration_rows = []
    
    for degradation in ["blur", "noise", "low_light"]:
        for severity in range(1, 6):
            experiment_conditions.append((degradation, severity))
    
    for degradation, severity in experiment_conditions:
        print(f"Evaluating {degradation}, severity {severity}")

        rows = evaluate_condition(
            model=model,
            device=device,
            batch_size=args.batch_size,
            degradation=degradation,
            severity=severity,
            max_eval_batches=args.max_eval_batches
        )
        all_prediction_rows.extend(rows)
        all_metric_rows.append(summarise_condition(rows))
        correct = [row["correct"] for row in rows]
        confidences = [row["confidence"] for row in rows]

        for bin_row in calibration_bins(correct, confidences, n_bins=10):
            bin_row["dataset"] = "MNIST"
            bin_row["model"] = "SimpleCNN"
            bin_row["degradation"] = degradation
            bin_row["severity"] = severity
            all_calibration_rows.append(bin_row)

    predictions_df = pd.DataFrame(all_prediction_rows)
    metrics_df = pd.DataFrame(all_metric_rows)
    calibration_df = pd.DataFrame(all_calibration_rows)

    predictions_df.to_csv(output_path / "predictions.csv", index=False)
    metrics_df.to_csv(output_path / "metrics_summary.csv", index=False)
    calibration_df.to_csv(output_path / "calibration_bins.csv", index=False)
    
    print(f"Saved predictions to {output_path / 'predictions.csv'}")
    print(f"Saved metrics to {output_path / 'metrics_summary.csv'}")
    print(f"Saved calibration bins to {output_path / 'calibration_bins.csv'}")

if __name__ == "__main__":
    main()