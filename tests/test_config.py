"""Tests for experiment configuration."""

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