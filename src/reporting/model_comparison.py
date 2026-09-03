"""Compare saved model reliability results."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
from src.reporting.evidence_analysis import build_confidence_diagnostics, build_prediction_confidence_transitions

plt.switch_backend("Agg")

PLOT_DPI = 200
COMPARISON_METRICS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "mean_confidence": "Mean confidence",
    "confidence_accuracy_gap": "Confidence-accuracy gap",
    "ece": "Expected calibration error",
    "hcer_fixed": "Fixed high-confidence error rate"
}
PAIR_COLUMNS = [
    "dataset", "seed", "degradation", "severity", "validation_split",
    "requested_validation_size", "validation_size", "validation_track_count",
    "track_overlap", "validation_track_hash", "fixed_hcer_threshold",
    "rank_hcer_top_fraction", "adaptive_hcer_percentile", "ece_bins",
    "num_examples"
]

def _single_value(metrics_df: pd.DataFrame, column: str):
    values = metrics_df[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"Expected one {column} value per metrics file.")
    return values.iloc[0]

# make sure both runs used the exact same split, track hash, and conditions before comparing
def _validate_metrics_pair(first_df: pd.DataFrame, second_df: pd.DataFrame) -> tuple[str, str]:
    required = set(PAIR_COLUMNS) | {"model"} | set(COMPARISON_METRICS)

    for metrics_df in (first_df, second_df):
        missing = required.difference(metrics_df.columns)
        if missing:
            raise ValueError(f"Missing comparison columns: {', '.join(sorted(missing))}.")
        if len(metrics_df[["degradation", "severity"]].drop_duplicates()) != len(metrics_df):
            raise ValueError("Each metrics file must contain one row per condition.")

    first_model = str(_single_value(first_df, "model"))
    second_model = str(_single_value(second_df, "model"))

    if first_model == second_model:
        raise ValueError("Model comparison requires two different models.")

    first_conditions = set(map(tuple, first_df[["degradation", "severity"]].to_numpy()))
    second_conditions = set(map(tuple, second_df[["degradation", "severity"]].to_numpy()))

    if first_conditions != second_conditions:
        raise ValueError("Model comparison requires identical degradation conditions.")

    for column in [name for name in PAIR_COLUMNS if name not in {"degradation", "severity"}]:
        first_values = first_df.sort_values(["degradation", "severity"])[column].reset_index(drop=True)
        second_values = second_df.sort_values(["degradation", "severity"])[column].reset_index(drop=True)

        if not first_values.equals(second_values):
            raise ValueError(f"Model comparison requires matching {column} values.")

    return first_model, second_model

def build_model_comparison(first_df: pd.DataFrame, second_df: pd.DataFrame) -> pd.DataFrame:
    """Join two matching model evaluations condition by condition."""
    first_model, second_model = _validate_metrics_pair(first_df, second_df)
    key_columns = ["dataset", "seed", "degradation", "severity", "validation_split", "validation_track_hash", "num_examples"]
    first = first_df[key_columns + list(COMPARISON_METRICS)].copy()
    second = second_df[key_columns + list(COMPARISON_METRICS)].copy()

    first = first.rename(columns={metric: f"{first_model}_{metric}" for metric in COMPARISON_METRICS})
    second = second.rename(columns={metric: f"{second_model}_{metric}" for metric in COMPARISON_METRICS})
    comparison = first.merge(second, on=key_columns, how="inner", validate="one_to_one")

    # keep signed differences so it's obvious whether accuracy improved or errors/ece went up
    for metric in COMPARISON_METRICS:
        comparison[f"{metric}_delta_{second_model}_minus_{first_model}"] = comparison[f"{second_model}_{metric}"] - comparison[f"{first_model}_{metric}"]

    order = {"none": 0, "blur": 1, "noise": 2, "low_light": 3}
    comparison["_degradation_order"] = comparison["degradation"].map(order).fillna(99)

    return comparison.sort_values(["_degradation_order", "severity"]).drop(columns="_degradation_order").reset_index(drop=True)

def build_trust_transition_comparison(first_records: list[dict], second_records: list[dict]) -> pd.DataFrame:
    """Compare the first trust warnings produced by two models."""
    first_df = pd.DataFrame(first_records)
    second_df = pd.DataFrame(second_records)
    required = {"dataset", "model", "degradation", "severity", "trust_signal"}

    for trust_df in (first_df, second_df):
        missing = required.difference(trust_df.columns)
        if missing:
            raise ValueError(f"Missing trust columns: {', '.join(sorted(missing))}.")
        if len(trust_df[["degradation", "severity"]].drop_duplicates()) != len(trust_df):
            raise ValueError("Each trust file must contain one row per condition.")

    first_model = str(_single_value(first_df, "model"))
    second_model = str(_single_value(second_df, "model"))

    if first_model == second_model:
        raise ValueError("Trust comparison requires two different models.")

    first_conditions = set(map(tuple, first_df[["degradation", "severity"]].to_numpy()))
    second_conditions = set(map(tuple, second_df[["degradation", "severity"]].to_numpy()))

    if first_conditions != second_conditions:
        raise ValueError("Trust comparison requires identical degradation conditions.")

    rows = []
    degradations = first_df.loc[first_df["degradation"] != "none", "degradation"].drop_duplicates().tolist()

    for degradation in degradations:
        row = {"degradation": degradation}

        for model_name, trust_df in ((first_model, first_df), (second_model, second_df)):
            condition = trust_df[trust_df["degradation"] == degradation]
            caution = condition.loc[condition["trust_signal"] == "caution", "severity"]
            do_not_trust = condition.loc[condition["trust_signal"] == "do_not_trust", "severity"]
            row[f"{model_name}_first_caution_severity"] = int(caution.min()) if not caution.empty else None
            row[f"{model_name}_first_do_not_trust_severity"] = int(do_not_trust.min()) if not do_not_trust.empty else None

        rows.append(row)

    return pd.DataFrame(rows)

def _first_warning_summary(trust_df: pd.DataFrame, degradation: str) -> str:
    condition = trust_df[trust_df["degradation"] == degradation].sort_values("severity")
    if condition.empty:
        raise ValueError(f"Missing trust records for {degradation}.")

    warning = condition[condition["trust_signal"] != "trust"]
    if warning.empty:
        return "none"

    first = warning.iloc[0]
    channels = []

    if first["performance_signal"] != "trust":
        channels.append("performance")
    if first["confidence_signal"] != "trust":
        channels.append("confidence")
    if not channels:
        raise ValueError(f"{degradation} warning has no active warning channel.")

    return f"{first['trust_signal']}@{int(first['severity'])} ({'+'.join(channels)})"

def build_confirmatory_model_summary(model_evidence: list[tuple[pd.DataFrame, pd.DataFrame, list[dict]]]) -> pd.DataFrame:
    """Summarise the selected evidence used to check findings across models."""
    if len(model_evidence) < 2:
        raise ValueError("Confirmatory comparison requires at least two models.")

    reference_metrics = model_evidence[0][0]
    for metrics, _, _ in model_evidence[1:]:
        _validate_metrics_pair(reference_metrics, metrics)

    rows = []

    for metrics, predictions, trust_records in model_evidence:
        model = str(_single_value(metrics, "model"))
        diagnostics = build_confidence_diagnostics(predictions, metrics)
        transitions = build_prediction_confidence_transitions(predictions)
        trust_df = pd.DataFrame(trust_records)
        required_trust_columns = {
            "model", "degradation", "severity", "trust_signal",
            "performance_signal", "confidence_signal"
        }
        missing = required_trust_columns.difference(trust_df.columns)

        if missing:
            raise ValueError(f"Missing trust columns: {', '.join(sorted(missing))}.")
        if trust_df["model"].drop_duplicates().tolist() != [model]:
            raise ValueError("Trust records and metrics describe different models.")
        if trust_df.duplicated(["degradation", "severity"]).any():
            raise ValueError("Trust records must contain one row per condition.")

        clean = metrics[(metrics["degradation"] == "none") & (metrics["severity"] == 0)]
        severe_noise = metrics[(metrics["degradation"] == "noise") & (metrics["severity"] == 5)]
        severe_noise_diagnostics = diagnostics[
            (diagnostics["degradation"] == "noise")
            & (diagnostics["severity"] == 5)
        ]
        persistent_errors = transitions[
            (transitions["degradation"] == "noise")
            & (transitions["from_severity"] == 4)
            & (transitions["to_severity"] == 5)
        ]

        if len(clean) != 1 or len(severe_noise) != 1:
            raise ValueError("Confirmatory comparison requires clean and noise severity 5 metrics.")
        if len(severe_noise_diagnostics) != 1:
            raise ValueError("Confirmatory comparison requires noise severity 5 confidence diagnostics.")
        if len(persistent_errors) != 1:
            raise ValueError("Confirmatory comparison requires the noise severity 4 to 5 transition.")

        clean = clean.iloc[0]
        severe_noise = severe_noise.iloc[0]
        severe_noise_diagnostics = severe_noise_diagnostics.iloc[0]
        persistent_errors = persistent_errors.iloc[0]

        rows.append({
            "model": model,
            "clean_test_accuracy": float(clean["accuracy"]),
            "clean_test_balanced_accuracy": float(clean["balanced_accuracy"]),
            "noise_5_accuracy": float(severe_noise["accuracy"]),
            "noise_5_balanced_accuracy": float(severe_noise["balanced_accuracy"]),
            "noise_5_failure_detection_auroc": float(severe_noise_diagnostics["failure_detection_auroc"]),
            "noise_5_rank_high_confidence_accuracy": float(severe_noise_diagnostics["rank_high_confidence_accuracy"]),
            "noise_4_to_5_persistent_error_count": int(persistent_errors["wrong_at_both_count"]),
            "noise_4_to_5_persistent_error_confidence_increased_rate": float(persistent_errors["wrong_at_both_confidence_increased_rate"]),
            "noise_4_to_5_persistent_error_mean_confidence_change": float(persistent_errors["wrong_at_both_mean_confidence_change"]),
            "blur_first_warning": _first_warning_summary(trust_df, "blur"),
            "noise_first_warning": _first_warning_summary(trust_df, "noise"),
            "low_light_first_warning": _first_warning_summary(trust_df, "low_light"),
            "dataset": str(clean["dataset"]),
            "seed": int(clean["seed"]),
            "test_images": int(clean["num_examples"]),
            "rank_hcer_top_fraction": float(clean["rank_hcer_top_fraction"]),
            "validation_split": str(clean["validation_split"]),
            "validation_track_hash": str(clean["validation_track_hash"]),
            "track_overlap": int(clean["track_overlap"])
        })

    return pd.DataFrame(rows)

def _condition_frame(metrics_df: pd.DataFrame, degradation: str) -> pd.DataFrame:
    baseline = metrics_df[(metrics_df["degradation"] == "none") & (metrics_df["severity"] == 0)].copy()
    condition = metrics_df[metrics_df["degradation"] == degradation].copy()

    if len(baseline) != 1 or condition.empty:
        raise ValueError(f"Missing baseline or degradation rows for {degradation}.")

    # reuse the same baseline so each degradation comparison curve starts at severity zero
    baseline["degradation"] = degradation
    return pd.concat([baseline, condition], ignore_index=True).sort_values("severity")

def save_model_comparison_plots(first_df: pd.DataFrame, second_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save direct metric comparisons for two matching models."""
    first_model, second_model = _validate_metrics_pair(first_df, second_df)
    output_dir.mkdir(parents=True, exist_ok=True)
    degradations = first_df.loc[first_df["degradation"] != "none", "degradation"].drop_duplicates().tolist()
    saved_paths = []

    for metric, ylabel in COMPARISON_METRICS.items():
        figure, axes = plt.subplots(1, len(degradations), figsize=(13, 4), sharey=True)

        if len(degradations) == 1:
            axes = [axes]

        for axis, degradation in zip(axes, degradations):
            first_condition = _condition_frame(first_df, degradation)
            second_condition = _condition_frame(second_df, degradation)
            axis.plot(first_condition["severity"], first_condition[metric], marker="o", label=first_model)
            axis.plot(second_condition["severity"], second_condition[metric], marker="o", linestyle="--", label=second_model)
            axis.set_title(degradation.replace("_", " ").title())
            axis.set_xlabel("Severity")
            axis.set_xticks(range(int(first_condition["severity"].max()) + 1))
            axis.grid(alpha=0.3)

        axes[0].set_ylabel(ylabel)

        if metric == "confidence_accuracy_gap":
            for axis in axes:
                axis.axhline(0.0, linewidth=1, linestyle=":")
        elif metric == "hcer_fixed":
            for axis in axes:
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                axis.set_ylim(bottom=0)
        else:
            for axis in axes:
                axis.set_ylim(bottom=0)

        axes[-1].legend()
        figure.suptitle(f"{first_df.iloc[0]['dataset']}: {ylabel} by model")
        figure.tight_layout(rect=(0, 0, 1, 0.93))

        output_path = output_dir / f"{metric}_model_comparison.png"
        figure.savefig(output_path, dpi=PLOT_DPI)
        plt.close(figure)
        saved_paths.append(output_path)

    return saved_paths