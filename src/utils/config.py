"""Load and validate experiment configuration."""

from __future__ import annotations
from pathlib import Path
import shutil
import yaml
from src.models.gtsrb_models import SUPPORTED_GTSRB_MODELS, get_supported_gtsrb_pretrained_weights

SUPPORTED_DEGRADATIONS = {"blur", "noise", "low_light"}
SUPPORTED_GTSRB_VALIDATION_SPLITS = {"stratified_track"}
SUPPORTED_GTSRB_TRAINING_STRATEGIES = {"from_scratch", "full_finetune"}
SUPPORTED_DATASET_MODELS = {
    "MNIST": {"SimpleCNN"},
    "GTSRB": SUPPORTED_GTSRB_MODELS
}

AUGMENTATION_KEYS = (
    "resize",
    "random_crop",
    "rotation",
    "blur",
    "noise",
    "brightness_contrast"
)
TRUST_METRICS = (
    "absolute_accuracy_drop",
    "relative_error_increase",
    "ece_increase",
    "gap_deterioration",
    "fixed_hcer_increase",
    "adaptive_hcer_increase"
)

def _require_non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")

def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")

def _require_positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")

def _require_optional_positive_int(value: object, name: str) -> None:
    if value is not None:
        _require_positive_int(value, name)

def _require_number_in_range(value: object, name: str, minimum: float, maximum: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")

def _validate_gtsrb_model_config(model: str, training: dict) -> None:

    pretrained_weights = training.get("pretrained_weights")
    if pretrained_weights is not None and not isinstance(pretrained_weights, str):
        raise ValueError("training.pretrained_weights must be a string or null.")
    if pretrained_weights not in get_supported_gtsrb_pretrained_weights(model):
        raise ValueError(f"Unsupported pretrained weights for {model}.")

    training_strategy = training.get("training_strategy")
    # default to from_scratch so older baseline configs without this key still work
    if model == "GTSRBCNN" and training_strategy is None:
        training_strategy = "from_scratch"
    if training_strategy not in SUPPORTED_GTSRB_TRAINING_STRATEGIES:
        raise ValueError("training.training_strategy must be from_scratch or full_finetune.")
    # prevent invalid combinations: random weights cannot be fine-tuned, and pretrained weights must not train from scratch
    if pretrained_weights is None and training_strategy != "from_scratch":
        raise ValueError("Models without pretrained weights must use from_scratch training.")
    if pretrained_weights is not None and training_strategy != "full_finetune":
        raise ValueError("Pretrained models must use full_finetune training.")
    
def _validate_trust_thresholds(thresholds: object, name: str, metrics: list[str] | tuple[str, ...]) -> None:
    if not isinstance(thresholds, dict):
        raise ValueError(f"{name} must be a mapping.")

    for metric in metrics:
        value = thresholds.get(metric)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name}.{metric} must be a non-negative number.")

def _validate_trust_policy(policy: object) -> None:
    if not isinstance(policy, dict):
        raise ValueError("trust_policy must be a mapping.")

    error_floor = policy.get("error_denominator_floor")
    # the floor prevents near-zero baseline error from making the ratio explode
    if isinstance(error_floor, bool) or not isinstance(error_floor, (int, float)) or not 0 < error_floor <= 1:
        raise ValueError("trust_policy.error_denominator_floor must be greater than 0 and no greater than 1.")

    active_metrics = policy.get("active_metrics", TRUST_METRICS)
    if not isinstance(active_metrics, (list, tuple)) or not active_metrics:
        raise ValueError("trust_policy.active_metrics must be a non-empty list.")
    if any(metric not in TRUST_METRICS for metric in active_metrics):
        raise ValueError("trust_policy.active_metrics contains an unsupported metric.")
    if len(active_metrics) != len(set(active_metrics)):
        raise ValueError("trust_policy.active_metrics must not contain duplicates.")

    caution = policy.get("caution")
    do_not_trust = policy.get("do_not_trust")
    _validate_trust_thresholds(caution, "trust_policy.caution", active_metrics)
    _validate_trust_thresholds(do_not_trust, "trust_policy.do_not_trust", active_metrics)

    # stronger warnings must not be easier to trigger than caution warnings
    for metric in active_metrics:
        if do_not_trust[metric] < caution[metric]:
            raise ValueError(f"trust_policy.do_not_trust.{metric} must be at least the caution threshold.")
        
def validate_config(config: dict) -> None:
    for key in ("dataset", "model", "data_dir", "checkpoint", "output_dir", "validation_profile"):
        _require_non_empty_string(config.get(key), key)

    dataset = config["dataset"]
    model = config["model"]

    if dataset not in SUPPORTED_DATASET_MODELS:
        raise ValueError(f"Unsupported dataset: {dataset}.")
    if model not in SUPPORTED_DATASET_MODELS[dataset]:
        raise ValueError(f"Unsupported model for {dataset}: {model}.")

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
    _require_optional_positive_int(
        training.get("max_train_batches"),
        "training.max_train_batches"
    )

    if config["dataset"] == "GTSRB":
        validation_split = training.get("validation_split")
        if validation_split not in SUPPORTED_GTSRB_VALIDATION_SPLITS:
            raise ValueError(
                "training.validation_split must be stratified_track for GTSRB."
            )
        _validate_gtsrb_model_config(config["model"], training)
        
    learning_rate = training.get("learning_rate")
    if learning_rate is not None:
        _require_positive_number(learning_rate, "training.learning_rate")

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

    if config["dataset"] == "GTSRB":
        rank_fraction = evaluation.get("rank_hcer_top_fraction")
        if (
            isinstance(rank_fraction, bool)
            or not isinstance(rank_fraction, (int, float))
            or not 0 < rank_fraction <= 1
        ):
            raise ValueError("evaluation.rank_hcer_top_fraction must be greater than 0 and no greater than 1.")

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
    
    _validate_trust_policy(config.get("trust_policy"))

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