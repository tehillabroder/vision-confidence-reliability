"""Tests for track-aware GTSRB splitting."""

from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.gtsrb import GTSRBDataset
from src.datasets.gtsrb_split import (
    GTSRB_TRACK_SPLIT,
    extract_gtsrb_track_id,
    split_gtsrb_by_track,
    validate_gtsrb_split_metadata
)

class FakeTrackDataset(Dataset):
    """Provide three classes with two complete tracks each."""

    def __init__(self, root: Path):
        self._samples = []
        self.labels = []

        for label in (0, 1, 42):
            for track in range(2):
                for frame in range(2):
                    image_path = (
                        root
                        / f"{label:05d}"
                        / f"{track:05d}_{frame:05d}.ppm"
                    )
                    self._samples.append((str(image_path), label))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return Image.new("RGB", (16, 16)), self.labels[index]

def build_dataset(tmp_path) -> GTSRBDataset:
    return GTSRBDataset(FakeTrackDataset(tmp_path))

def subset_track_ids(dataset: GTSRBDataset, indices: list[int]) -> set[str]:
    samples = dataset.base_dataset._samples
    return {
        extract_gtsrb_track_id(samples[index][0], samples[index][1])
        for index in indices
    }

def test_extract_gtsrb_track_id_combines_class_and_track():
    # confirm repeated track numbers in different classes remain separate
    track_id = extract_gtsrb_track_id(
        "00042/00003_00017.ppm",
        42
    )

    assert track_id == "42:00003"

def test_track_split_is_repeatable_and_has_no_overlap(tmp_path):
    # confirm the same seed selects the same complete validation tracks
    dataset = build_dataset(tmp_path)

    first_train, first_validation, first_metadata = split_gtsrb_by_track(
        dataset,
        validation_size=6,
        seed=42,
        split_strategy=GTSRB_TRACK_SPLIT
    )
    second_train, second_validation, second_metadata = split_gtsrb_by_track(
        dataset,
        validation_size=6,
        seed=42,
        split_strategy=GTSRB_TRACK_SPLIT
    )

    training_tracks = subset_track_ids(
        dataset,
        first_train.indices
    )
    validation_tracks = subset_track_ids(
        dataset,
        first_validation.indices
    )

    assert first_train.indices == second_train.indices
    assert first_validation.indices == second_validation.indices
    assert first_metadata == second_metadata
    assert training_tracks.isdisjoint(validation_tracks)
    assert first_metadata["track_overlap"] == 0
    assert first_metadata["training_class_count"] == 3
    assert first_metadata["validation_class_count"] == 3

def test_track_split_keeps_every_track_complete(tmp_path):
    # ensure no track is divided between training and validation
    dataset = build_dataset(tmp_path)
    train_set, validation_set, metadata = split_gtsrb_by_track(
        dataset,
        validation_size=6,
        seed=42,
        split_strategy=GTSRB_TRACK_SPLIT
    )

    assert len(train_set) == 6
    assert len(validation_set) == 6
    assert metadata["train_track_count"] == 3
    assert metadata["validation_track_count"] == 3
    assert metadata["track_size"] == 2

def test_track_split_uses_nearest_complete_track_size(tmp_path):
    # check that a target size is rounded to complete tracks
    dataset = build_dataset(tmp_path)
    _, validation_set, metadata = split_gtsrb_by_track(
        dataset,
        validation_size=5,
        seed=42,
        split_strategy=GTSRB_TRACK_SPLIT
    )

    assert len(validation_set) == 6
    assert metadata["requested_validation_size"] == 5
    assert metadata["validation_size_difference"] == 1

def test_track_split_rejects_unknown_strategy(tmp_path):
    # ensure unsupported split strategies fail clearly
    dataset = build_dataset(tmp_path)

    with pytest.raises(
        ValueError,
        match="Unsupported GTSRB validation split"
    ):
        split_gtsrb_by_track(
            dataset,
            validation_size=6,
            seed=42,
            split_strategy="random"
        )

def test_track_id_rejects_unrecognised_filename():
    # ensure malformed filenames cannot silently create wrong groups
    with pytest.raises(ValueError, match="Cannot extract GTSRB track ID"):
        extract_gtsrb_track_id(
            "traffic_sign.ppm",
            1
        )

def test_track_split_metadata_has_valid_fingerprint(tmp_path):
    # confirm split evidence includes a repeatable track fingerprint
    dataset = build_dataset(tmp_path)
    _, _, metadata = split_gtsrb_by_track(
        dataset,
        validation_size=6,
        seed=42,
        split_strategy=GTSRB_TRACK_SPLIT
    )

    validated = validate_gtsrb_split_metadata(
        metadata=metadata,
        validation_split=GTSRB_TRACK_SPLIT,
        requested_validation_size=6,
        class_count=3
    )

    assert len(validated["validation_track_hash"]) == 64