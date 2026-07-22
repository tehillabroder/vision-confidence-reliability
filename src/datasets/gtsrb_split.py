"""Track-aware GTSRB split helpers."""

from collections import defaultdict
from math import floor
from pathlib import Path
from hashlib import sha256

import torch
from torch.utils.data import Dataset, Subset

GTSRB_TRACK_SPLIT = "stratified_track"

def extract_gtsrb_track_id(image_path: str, label: int) -> str:
    """Extract a class-specific track identifier."""
    filename_parts = Path(image_path).stem.split("_")

    if (
        len(filename_parts) != 2
        or not filename_parts[0].isdigit()
        or not filename_parts[1].isdigit()
    ):
        raise ValueError(f"Cannot extract GTSRB track ID from: {image_path}")

    return f"{label}:{filename_parts[0]}"

def _read_gtsrb_samples(dataset: Dataset) -> list[tuple[str, int]]:
    base_dataset = getattr(dataset, "base_dataset", None)
    samples = getattr(base_dataset, "_samples", None)

    if not isinstance(samples, list) or len(samples) != len(dataset):
        raise ValueError("GTSRB track splitting requires the torchvision sample paths.")

    parsed_samples = []

    for sample in samples:
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("Each GTSRB sample must contain an image path and label.")

        image_path, label = sample
        parsed_samples.append((str(image_path), int(label)))

    return parsed_samples

def _allocate_validation_tracks(
    class_track_counts: dict[int, int],
    validation_track_count: int
) -> dict[int, int]:
    labels = sorted(class_track_counts)
    total_track_count = sum(class_track_counts.values())

    if validation_track_count < len(labels):
        raise ValueError(
            "Validation size is too small to include one track from every class."
        )

    maximum_validation_tracks = sum(
        track_count - 1 for track_count in class_track_counts.values()
    )
    if validation_track_count > maximum_validation_tracks:
        raise ValueError(
            "Validation size must leave at least one training track in every class."
        )

    raw_allocations = {
        label: class_track_counts[label] * validation_track_count / total_track_count
        for label in labels
    }
    allocations = {
        label: min(
            class_track_counts[label] - 1,
            max(1, floor(raw_allocations[label]))
        )
        for label in labels
    }

    while sum(allocations.values()) < validation_track_count:
        candidates = [
            label
            for label in labels
            if allocations[label] < class_track_counts[label] - 1
        ]
        label = max(
            candidates,
            key=lambda item: (
                raw_allocations[item] - allocations[item],
                class_track_counts[item],
                -item
            )
        )
        allocations[label] += 1

    while sum(allocations.values()) > validation_track_count:
        candidates = [
            label
            for label in labels
            if allocations[label] > 1
        ]
        label = max(
            candidates,
            key=lambda item: (
                allocations[item] - raw_allocations[item],
                -class_track_counts[item],
                item
            )
        )
        allocations[label] -= 1

    return allocations

def split_gtsrb_by_track(
    dataset: Dataset,
    validation_size: int,
    seed: int,
    split_strategy: str
) -> tuple[Subset, Subset, dict[str, object]]:
    """Split GTSRB while keeping complete tracks together."""
    if split_strategy != GTSRB_TRACK_SPLIT:
        raise ValueError(
            f"Unsupported GTSRB validation split: {split_strategy}"
        )
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError(
            "Validation size must be greater than zero and smaller than the dataset."
        )

    samples = _read_gtsrb_samples(dataset)
    track_indices = defaultdict(list)
    class_tracks = defaultdict(set)

    for index, (image_path, label) in enumerate(samples):
        track_id = extract_gtsrb_track_id(image_path, label)
        track_indices[track_id].append(index)
        class_tracks[label].add(track_id)

    track_sizes = {len(indices) for indices in track_indices.values()}
    if len(track_sizes) != 1:
        raise ValueError("GTSRB tracks must contain a consistent number of images.")

    track_size = track_sizes.pop()
    validation_track_count = (validation_size + track_size // 2) // track_size
    class_track_counts = {
        label: len(track_ids)
        for label, track_ids in class_tracks.items()
    }
    allocations = _allocate_validation_tracks(
        class_track_counts,
        validation_track_count
    )

    generator = torch.Generator().manual_seed(seed)
    validation_tracks = set()

    for label in sorted(class_tracks):
        track_ids = sorted(class_tracks[label])
        order = torch.randperm(
            len(track_ids),
            generator=generator
        ).tolist()
        shuffled_tracks = [track_ids[index] for index in order]
        validation_tracks.update(
            shuffled_tracks[:allocations[label]]
        )

    all_tracks = set(track_indices)
    training_tracks = all_tracks - validation_tracks
    overlap = training_tracks & validation_tracks

    training_indices = sorted(
        index
        for track_id in training_tracks
        for index in track_indices[track_id]
    )
    validation_indices = sorted(
        index
        for track_id in validation_tracks
        for index in track_indices[track_id]
    )

    validation_track_hash = sha256(
        "\n".join(sorted(validation_tracks)).encode("utf-8")
    ).hexdigest()

    metadata = {
        "validation_split": split_strategy,
        "requested_validation_size": validation_size,
        "validation_size": len(validation_indices),
        "validation_size_difference": len(validation_indices) - validation_size,
        "train_size": len(training_indices),
        "track_size": track_size,
        "total_track_count": len(all_tracks),
        "train_track_count": len(training_tracks),
        "validation_track_count": len(validation_tracks),
        "track_overlap": len(overlap),
        "training_class_count": len({
            samples[index][1] for index in training_indices
        }),
        "validation_class_count": len({
            samples[index][1] for index in validation_indices
        }),
        "validation_track_hash": validation_track_hash
    }

    return (
        Subset(dataset, training_indices),
        Subset(dataset, validation_indices),
        metadata
    )

def validate_gtsrb_split_metadata(
    metadata: object,
    validation_split: str,
    requested_validation_size: int,
    class_count: int
) -> dict[str, object]:
    """Validate saved GTSRB split evidence."""
    if not isinstance(metadata, dict):
        raise ValueError("GTSRB split metadata must be a mapping.")

    required_keys = {
        "validation_split",
        "requested_validation_size",
        "validation_size",
        "validation_size_difference",
        "train_size",
        "track_size",
        "total_track_count",
        "train_track_count",
        "validation_track_count",
        "track_overlap",
        "training_class_count",
        "validation_class_count",
        "validation_track_hash"
    }
    missing_keys = sorted(required_keys - metadata.keys())

    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"GTSRB split metadata is missing: {missing}")

    expected_values = {
        "validation_split": validation_split,
        "requested_validation_size": requested_validation_size,
        "training_class_count": class_count,
        "validation_class_count": class_count,
        "track_overlap": 0
    }

    for name, expected_value in expected_values.items():
        if metadata.get(name) != expected_value:
            raise ValueError(
                f"GTSRB split metadata {name} does not match the configuration."
            )

    integer_keys = required_keys - {
        "validation_split",
        "validation_track_hash"
    }

    for name in integer_keys:
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"GTSRB split metadata {name} must be an integer.")

    if metadata["validation_size"] <= 0 or metadata["train_size"] <= 0:
        raise ValueError("GTSRB split subsets must not be empty.")

    expected_validation_size = (
        metadata["validation_track_count"] * metadata["track_size"]
    )
    if metadata["validation_size"] != expected_validation_size:
        raise ValueError(
            "GTSRB validation size does not match its track count."
        )

    expected_train_size = (
        metadata["train_track_count"] * metadata["track_size"]
    )
    if metadata["train_size"] != expected_train_size:
        raise ValueError(
            "GTSRB training size does not match its track count."
        )

    expected_track_count = (
        metadata["train_track_count"]
        + metadata["validation_track_count"]
    )
    if metadata["total_track_count"] != expected_track_count:
        raise ValueError("GTSRB total track count is inconsistent.")

    expected_difference = (
        metadata["validation_size"]
        - metadata["requested_validation_size"]
    )
    if metadata["validation_size_difference"] != expected_difference:
        raise ValueError(
            "GTSRB validation size difference is inconsistent."
        )

    track_hash = metadata["validation_track_hash"]
    valid_characters = set("0123456789abcdef")

    if (
        not isinstance(track_hash, str)
        or len(track_hash) != 64
        or any(character not in valid_characters for character in track_hash)
    ):
        raise ValueError(
            "GTSRB validation track hash must be a SHA-256 value."
        )

    return metadata