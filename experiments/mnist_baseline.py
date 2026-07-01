"""Minimal MNIST baseline prototype.

Loads MNIST, defines a simple CNN, trains briefly and reports clean test
accuracy. This is a scaffolding prototype for the "Confident and Wrong"
project -- not an attempt to train the best classifier. No large files
(e.g. model checkpoints) are saved; the MNIST data lives under data/,
which is gitignored.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "data"


def set_seed(seed: int):
    # Set torch seeds for repeatable runs.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleCNN(nn.Module):
    """A small two-conv-layer CNN for 28x28 grayscale digits."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, loader, device, epochs: int, max_train_batches=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        batches = 0
        for images, labels in loader:
            # Stop early when a batch cap is set.
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
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Minimal MNIST baseline")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs (brief by default)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42, help="fixed random seed")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="cap training batches per epoch for quick debugging")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(args.batch_size)
    print(f"Train samples {len(train_loader.dataset)}")
    print(f"Test samples {len(test_loader.dataset)}")
    model = SimpleCNN().to(device)

    train(model, train_loader, device, args.epochs, args.max_train_batches)
    accuracy = evaluate(model, test_loader, device)
    print(f"Clean test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
