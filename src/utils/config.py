"""Load and validate experiment configuration."""

from pathlib import Path
import shutil

import yaml
SUPPORTED_DEGRADATIONS = {"blur", "noise", "low_light"}
AUGMENTATION_KEYS = (
    "resize",
    "random_crop",
    "rotation",
    "blur",
    "noise",
    "brightness_contrast"
)

def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")

def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def _require_optional_positive_int(value: object, name: str) -> None:
    if value is not None:
        _require_positive_int(value, name)

def _require_number_in_range(value: object, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

def validate_config(config: dict) -> None:
    for key in ("dataset", "model", "data_dir", "checkpoint", "output_dir"):
        _require_non_empty_string(config.get(key), key)

    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")

    training = config.get("training")
    evaluation = config.get("evaluation")
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping.")
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation must be a mapping.")

    _require_positive_int(training.get("epochs"), "training.epochs")
    _require_positive_int(training.get("batch_size"), "training.batch_size")
    _require_positive_int(training.get("validation_size"), "training.validation_size")
    _require_optional_positive_int(training.get("max_train_batches"), "training.max_train_batches")

    augmentation = training.get("augmentation")
    if not isinstance(augmentation, dict):
        raise ValueError("training.augmentation must be a mapping.")
    for key in AUGMENTATION_KEYS:
        if not isinstance(augmentation.get(key), bool):
            raise ValueError(f"training.augmentation.{key} must be true or false.")

    _require_positive_int(evaluation.get("batch_size"), "evaluation.batch_size")
    _require_optional_positive_int(evaluation.get("max_eval_batches"), "evaluation.max_eval_batches")
    _require_positive_int(evaluation.get("ece_bins"), "evaluation.ece_bins")
    _require_number_in_range(
        evaluation.get("fixed_hcer_threshold"),
        "evaluation.fixed_hcer_threshold",
        0.0,
        1.0
    )
    _require_number_in_range(
        evaluation.get("adaptive_hcer_percentile"),
        "evaluation.adaptive_hcer_percentile",
        0.0,
        100.0
    )

    degradations = evaluation.get("degradations")
    if not isinstance(degradations, list) or not degradations:
        raise ValueError("evaluation.degradations must be a non-empty list.")
    if any(not isinstance(name, str) for name in degradations):
        raise ValueError("evaluation.degradations must contain names.")
    if any(name not in SUPPORTED_DEGRADATIONS for name in degradations):
        raise ValueError("evaluation.degradations contains an unsupported degradation.")
    if len(degradations) != len(set(degradations)):
        raise ValueError("evaluation.degradations must not contain duplicates.")

    severity_levels = evaluation.get("severity_levels")
    if not isinstance(severity_levels, list) or not severity_levels:
        raise ValueError("evaluation.severity_levels must be a non-empty list.")
    if any(
        isinstance(severity, bool)
        or not isinstance(severity, int)
        or not 1 <= severity <= 5
        for severity in severity_levels
    ):
        raise ValueError("evaluation.severity_levels must contain integers from 1 to 5.")
    if len(severity_levels) != len(set(severity_levels)):
        raise ValueError("evaluation.severity_levels must not contain duplicates.")

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {config_path}") from error

    if not isinstance(config, dict):
        raise ValueError("Config must contain a mapping.")

    validate_config(config)
    return config

def save_config_copy(config_path: Path, destination: Path) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, destination)
    return destination