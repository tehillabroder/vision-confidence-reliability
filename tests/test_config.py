"""Tests for experiment configuration."""

from __future__ import annotations
from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config, save_config_copy

def valid_config() -> dict:
    return {
        "dataset": "MNIST",
        "model": "SimpleCNN",
        "data_dir": "data",
        "checkpoint": "checkpoints/mnist_simple_cnn.pt",
        "output_dir": "results/mnist_degradation_eval",
        "validation_profile": "results/mnist_validation_profile.json",
        "seed": 42,
        "training": {
            "epochs": 1,
            "batch_size": 64,
            "validation_size": 5000,
            "max_train_batches": None,
            "learning_rate": 0.001,
            "augmentation": {
                "resize": False,
                "random_crop": False,
                "rotation": False,
                "blur": False,
                "noise": False,
                "brightness_contrast": False
            }
        },
        "evaluation": {
            "batch_size": 64,
            "max_eval_batches": None,
            "degradations": ["blur", "noise", "low_light"],
            "severity_levels": [1, 2, 3, 4, 5],
            "ece_bins": 10,
            "fixed_hcer_threshold": 0.90,
            "adaptive_hcer_percentile": 90
        },
        "trust_policy": {
            "error_denominator_floor": 0.01,
            "caution": {
                "absolute_accuracy_drop": 0.05,
                "relative_error_increase": 1.0,
                "ece_increase": 0.03,
                "gap_deterioration": 0.05,
                "fixed_hcer_increase": 0.02,
                "adaptive_hcer_increase": 0.02
            },
            "do_not_trust": {
                "absolute_accuracy_drop": 0.15,
                "relative_error_increase": 3.0,
                "ece_increase": 0.08,
                "gap_deterioration": 0.12,
                "fixed_hcer_increase": 0.07,
                "adaptive_hcer_increase": 0.07
            }
        }
    }

def valid_gtsrb_config(
    model: str = "GTSRBCNN",
    pretrained_weights: str | None = None,
    training_strategy: str = "from_scratch"
) -> dict:
    config = valid_config()
    config["dataset"] = "GTSRB"
    config["model"] = model
    config["training"]["learning_rate"] = 0.001
    config["training"]["validation_split"] = "stratified_track"
    config["training"]["pretrained_weights"] = pretrained_weights
    config["training"]["training_strategy"] = training_strategy
    config["training"]["augmentation"]["resize"] = True
    config["evaluation"]["rank_hcer_top_fraction"] = 0.10
    return config

def write_config(tmp_path: Path, config: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path

def test_load_config_returns_valid_configuration(tmp_path):
    # confirm a complete configuration loads successfully
    config_path = write_config(tmp_path, valid_config())
    config = load_config(config_path)

    assert config["dataset"] == "MNIST"
    assert config["evaluation"]["fixed_hcer_threshold"] == 0.90

def test_load_config_rejects_missing_file(tmp_path):
    # ensure a missing configuration fails clearly
    with pytest.raises(FileNotFoundError, match="Config not found"):
        load_config(tmp_path / "missing.yaml")

def test_load_config_rejects_missing_training_section(tmp_path):
    # ensure required workflow sections cannot be omitted
    config = valid_config()
    del config["training"]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="training must be a mapping"):
        load_config(config_path)

def test_load_config_rejects_invalid_hcer_threshold(tmp_path):
    # ensure the fixed threshold remains a valid probability
    config = valid_config()
    config["evaluation"]["fixed_hcer_threshold"] = 1.1
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="fixed_hcer_threshold"):
        load_config(config_path)

def test_load_config_rejects_invalid_adaptive_percentile(tmp_path):
    # ensure the future adaptive percentile stays within its valid range
    config = valid_config()
    config["evaluation"]["adaptive_hcer_percentile"] = 101
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="adaptive_hcer_percentile"):
        load_config(config_path)

def test_load_config_rejects_invalid_severity(tmp_path):
    # ensure severity zero remains reserved for the undegraded condition
    config = valid_config()
    config["evaluation"]["severity_levels"] = [0, 1, 2]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="integers from 1 to 5"):
        load_config(config_path)

def test_save_config_copy_preserves_configuration(tmp_path):
    # confirm the exact configuration is saved with experiment evidence
    config_path = write_config(tmp_path, valid_config())
    destination = tmp_path / "results" / "config.yaml"

    saved_path = save_config_copy(config_path, destination)

    assert saved_path == destination
    assert destination.read_text(encoding="utf-8") == config_path.read_text(encoding="utf-8")

def test_load_config_rejects_zero_error_floor(tmp_path):
    # ensure relative error growth always has a stable denominator
    config = valid_config()
    config["trust_policy"]["error_denominator_floor"] = 0
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="error_denominator_floor"):
        load_config(config_path)

def test_load_config_rejects_weaker_do_not_trust_threshold(tmp_path):
    # ensure the strongest warning cannot trigger before caution
    config = valid_config()
    config["trust_policy"]["do_not_trust"]["ece_increase"] = 0.02
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="must be at least"):
        load_config(config_path)

def test_load_config_rejects_invalid_learning_rate(tmp_path):
    # ensure a configured learning rate must be positive
    config = valid_config()
    config["training"]["learning_rate"] = 0
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="training.learning_rate"):
        load_config(config_path)

def test_load_config_rejects_missing_learning_rate(tmp_path):
    # training settings shouldn't silently fall back to a hidden learning rate
    config = valid_config()
    del config["training"]["learning_rate"]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="training.learning_rate"):
        load_config(config_path)

def test_load_config_rejects_unimplemented_mnist_augmentation(tmp_path):
    # make sure the config cannot claim a training augmentation that MNIST doesn't use
    config = valid_config()
    config["training"]["augmentation"]["blur"] = True
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="implemented MNIST training pipeline"):
        load_config(config_path)

def test_load_config_rejects_incorrect_gtsrb_resize_setting(tmp_path):
    # ensure config records the resize that the GTSRB pipeline actually applies
    config = valid_gtsrb_config()
    config["training"]["augmentation"]["resize"] = False
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="implemented GTSRB training pipeline"):
        load_config(config_path)

def test_load_config_rejects_unknown_augmentation_key(tmp_path):
    # ensure unused augmentation settings cannot be accepted silently
    config = valid_config()
    config["training"]["augmentation"]["horizontal_flip"] = False
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unsupported keys"):
        load_config(config_path)

def test_load_config_accepts_gtsrb_track_split(tmp_path):
    # confirm the supported GTSRB track strategy loads successfully
    config = valid_gtsrb_config()
    config_path = write_config(tmp_path, config)

    loaded = load_config(config_path)

    assert loaded["training"]["validation_split"] == "stratified_track"

def test_load_config_rejects_gtsrb_random_split(tmp_path):
    # ensure GTSRB cannot return to image-level random splitting
    config = valid_gtsrb_config()
    config["training"]["validation_split"] = "random"
    config_path = write_config(tmp_path, config)
    with pytest.raises(ValueError, match="stratified_track"):
        load_config(config_path)

@pytest.mark.parametrize("model", ["ResNet18", "MobileNetV2"])
def test_load_config_accepts_pretrained_gtsrb_model(tmp_path, model):
    # confirm stronger GTSRB models can declare reproducible pretraining
    config = valid_gtsrb_config(
        model=model,
        pretrained_weights="IMAGENET1K_V1",
        training_strategy="full_finetune"
    )
    config_path = write_config(tmp_path, config)
    loaded = load_config(config_path)

    assert loaded["model"] == model
    assert loaded["training"]["pretrained_weights"] == "IMAGENET1K_V1"
    assert loaded["training"]["training_strategy"] == "full_finetune"

def test_load_config_rejects_pretrained_model_without_full_finetuning(tmp_path):
    # pretraining strategy mustn't be recorded inconsistently
    config = valid_gtsrb_config(
        model="ResNet18",
        pretrained_weights="IMAGENET1K_V1",
        training_strategy="from_scratch"
    )
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="must use full_finetune"):
        load_config(config_path)

def test_load_config_accepts_selected_trust_metrics(tmp_path):
    # confirm a dataset can exclude an audited metric from its trust policy
    config = valid_config()
    config["trust_policy"]["active_metrics"] = [
        "absolute_accuracy_drop",
        "relative_error_increase",
        "ece_increase",
        "gap_deterioration",
        "fixed_hcer_increase"
    ]
    del config["trust_policy"]["caution"]["adaptive_hcer_increase"]
    del config["trust_policy"]["do_not_trust"]["adaptive_hcer_increase"]
    config_path = write_config(tmp_path, config)

    loaded = load_config(config_path)

    assert "adaptive_hcer_increase" not in loaded["trust_policy"]["active_metrics"]

def test_load_config_rejects_unsupported_trust_metric(tmp_path):
    # ensure unknown trust rules cannot be enabled silently
    config = valid_config()
    config["trust_policy"]["active_metrics"] = ["unknown_metric"]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="unsupported metric"):
        load_config(config_path)

def test_load_config_rejects_unknown_dataset(tmp_path):
    # ensure unsupported datasets fail before an experiment starts
    config = valid_config()
    config["dataset"] = "UnknownDataset"
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="Unsupported dataset"):
        load_config(config_path)

def test_load_config_rejects_model_for_wrong_dataset(tmp_path):
    # ensure a valid model name cannot be used with the wrong dataset
    config = valid_config()
    config["model"] = "ResNet18"
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="Unsupported model for MNIST"):
        load_config(config_path)

def test_load_config_rejects_unknown_gtsrb_pretrained_weights(tmp_path):
    # ensure config and model construction accept the same weight names
    config = valid_gtsrb_config(
        model="ResNet18",
        pretrained_weights="DEFAULT",
        training_strategy="full_finetune"
    )
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="Unsupported pretrained weights"):
        load_config(config_path)

@pytest.mark.parametrize("rank_fraction", [0, 1.1])
def test_load_config_rejects_invalid_rank_hcer_fraction(tmp_path, rank_fraction):
    # ensure the configured rank is a valid fraction
    config = valid_gtsrb_config()
    config["evaluation"]["rank_hcer_top_fraction"] = rank_fraction
    config_path = write_config(tmp_path, config)

    with pytest.raises(ValueError, match="rank_hcer_top_fraction"):
        load_config(config_path)