"""MNIST dataset helpers."""

import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms

from src.degradations.image_degradations import apply_degradation

# facts about the implemented MNIST pipeline
MNIST_IMAGE_SIZE = (28, 28)
MNIST_NORMALISE_MEAN = (0.1307,)
MNIST_NORMALISE_STD = (0.3081,)
MNIST_NORMALISE = transforms.Normalize(MNIST_NORMALISE_MEAN, MNIST_NORMALISE_STD)
MNIST_TRAINING_AUGMENTATION = {
    "resize": False,
    "random_crop": False,
    "rotation": False,
    "blur": False,
    "noise": False,
    "brightness_contrast": False
}

def split_dataset(dataset: Dataset, validation_size: int, seed: int) -> tuple[Subset, Subset]:
    """Split a dataset into training and validation sets."""
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError("Validation size must be greater than zero and smaller than the dataset.")

    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator
    )
    return train_set, validation_set

def build_mnist_train_validation_datasets(data_dir: str, validation_size: int, seed: int) -> tuple[Subset, Subset]:
    """Load and split the MNIST training data."""
    transform = transforms.Compose([transforms.ToTensor(), MNIST_NORMALISE])
    full_train_set = datasets.MNIST(
        data_dir,
        train=True,
        download=True,
        transform=transform
    )
    return split_dataset(full_train_set, validation_size, seed)

class DegradedMNIST(Dataset):
    """MNIST test data with degradation before normalisation."""

    def __init__(self, data_dir: str, degradation: str, severity: int):
        self.base_dataset = datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.degradation = degradation
        self.severity = severity

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.base_dataset[index]
        degraded_image = apply_degradation(image, self.degradation, self.severity)
        normalised_image = MNIST_NORMALISE(degraded_image)
        return normalised_image, label, index