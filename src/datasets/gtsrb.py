"""GTSRB dataset helpers."""

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision import datasets, transforms
from src.degradations.image_degradations import apply_degradation

GTSRB_CLASS_COUNT = 43
# 64 pixels keeps sign detail without making later model trials unnecessarily heavy
GTSRB_IMAGE_SIZE = (64, 64)
# imagenet statistics support later trials with standard pretrained vision models
GTSRB_NORMALISE = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
GTSRB_PREPROCESS = transforms.Compose([
    transforms.Resize(GTSRB_IMAGE_SIZE, antialias=True),
    transforms.ToTensor()
])

def prepare_gtsrb_image(
    image: Image.Image,
    degradation: str = "none",
    severity: int = 0,
    normalise: bool = True
) -> torch.Tensor:
    """Prepare one RGB GTSRB image."""
    image_tensor = GTSRB_PREPROCESS(image.convert("RGB"))
    degraded_image = apply_degradation(image_tensor, degradation, severity)

    if not normalise:
        return degraded_image

    return GTSRB_NORMALISE(degraded_image)

class GTSRBDataset(Dataset):
    """GTSRB data with controlled degradation and stable identifiers."""

    def __init__(
        self,
        base_dataset: Dataset,
        degradation: str = "none",
        severity: int = 0,
        normalise: bool = True
    ):
        self.base_dataset = base_dataset
        self.degradation = degradation
        self.severity = severity
        self.normalise = normalise

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.base_dataset[index]
        label = int(label)

        if not 0 <= label < GTSRB_CLASS_COUNT:
            raise ValueError("GTSRB labels must be integers from 0 to 42.")

        prepared_image = prepare_gtsrb_image(
            image=image,
            degradation=self.degradation,
            severity=self.severity,
            normalise=self.normalise
        )
        return prepared_image, label, index

def split_gtsrb_dataset(dataset: Dataset, validation_size: int, seed: int) -> tuple[Subset, Subset]:
    """Split GTSRB training data deterministically."""
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

def build_gtsrb_train_validation_datasets(
    data_dir: str,
    validation_size: int,
    seed: int,
    download: bool = True
) -> tuple[Subset, Subset]:
    """Load and split the official GTSRB training data."""
    full_train_set = datasets.GTSRB(root=data_dir, split="train", download=download)
    dataset = GTSRBDataset(full_train_set)
    return split_gtsrb_dataset(dataset, validation_size, seed)

def build_gtsrb_test_dataset(
    data_dir: str,
    degradation: str = "none",
    severity: int = 0,
    download: bool = True,
    normalise: bool = True
) -> GTSRBDataset:
    """Load the official GTSRB test data."""
    test_set = datasets.GTSRB(root=data_dir, split="test", download=download)
    return GTSRBDataset(
        base_dataset=test_set,
        degradation=degradation,
        severity=severity,
        normalise=normalise
    )