"""Create plots from saved reliability metrics."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

# save plots without opening a display window
plt.switch_backend("Agg")

PLOT_DPI = 200
REQUIRED_COLUMNS = {
    "dataset", "model", "degradation", "severity", "accuracy",
    "mean_confidence", "confidence_accuracy_gap", "ece", "hcer"
}

def _validate_metrics(metrics_df: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS.difference(metrics_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required metric columns: {missing}")

    baseline_rows = metrics_df[
        (metrics_df["degradation"] == "none") & (metrics_df["severity"] == 0)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one undegraded baseline row.")

    experiment_pairs = metrics_df[["dataset", "model"]].drop_duplicates()
    if len(experiment_pairs) != 1:
        raise ValueError("Expected one dataset and model per metrics file.")

def _get_degradations(metrics_df: pd.DataFrame) -> list[str]:
    degradations = metrics_df.loc[
        metrics_df["degradation"] != "none", "degradation"
    ].drop_duplicates().tolist()

    if not degradations:
        raise ValueError("No degraded conditions were found.")

    return degradations

def _condition_frame(metrics_df: pd.DataFrame, degradation: str) -> pd.DataFrame:
    baseline = metrics_df[
        (metrics_df["degradation"] == "none") & (metrics_df["severity"] == 0)
    ].copy()
    condition = metrics_df[metrics_df["degradation"] == degradation].copy()

    if condition.empty:
        raise ValueError(f"No rows found for degradation: {degradation}")

    # reuse the undegraded result so every line starts at severity 0
    baseline["degradation"] = degradation

    return pd.concat([baseline, condition], ignore_index=True).sort_values("severity")

def _save_accuracy_confidence_plot(condition_df: pd.DataFrame, degradation: str, output_dir: Path) -> Path:
    dataset = condition_df.iloc[0]["dataset"]
    model = condition_df.iloc[0]["model"]

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        condition_df["severity"],
        condition_df["accuracy"],
        marker="o",
        label="Accuracy"
    )
    axis.plot(
        condition_df["severity"],
        condition_df["mean_confidence"],
        marker="o",
        label="Mean confidence"
    )

    axis.set_title(f"{dataset} {model}: {degradation.replace('_', ' ')}")
    axis.set_xlabel("Severity")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1)
    axis.set_xticks(range(int(condition_df["severity"].max()) + 1))
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()

    output_path = output_dir / f"{degradation}_accuracy_and_confidence.png"
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)

    return output_path

def _save_metric_plot(
    metrics_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
    output_dir: Path
) -> Path:
    dataset = metrics_df.iloc[0]["dataset"]
    model = metrics_df.iloc[0]["model"]
    degradations = _get_degradations(metrics_df)

    figure, axis = plt.subplots(figsize=(7, 4.5))

    for degradation in degradations:
        condition_df = _condition_frame(metrics_df, degradation)
        axis.plot(
            condition_df["severity"],
            condition_df[metric],
            marker="o",
            label=degradation.replace("_", " ").title()
        )

    if metric == "confidence_accuracy_gap":
        # zero separates overconfidence from underconfidence
        axis.axhline(0.0, linewidth=1, linestyle="--")
    elif metric == "hcer":
        # show the small error rates as percentages
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylim(bottom=0)
    else:
        axis.set_ylim(bottom=0)

    axis.set_title(f"{dataset} {model}: {ylabel}")
    axis.set_xlabel("Severity")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(int(metrics_df["severity"].max()) + 1))
    axis.legend(title="Degradation")
    axis.grid(alpha=0.3)
    figure.tight_layout()

    output_path = output_dir / filename
    figure.savefig(output_path, dpi=PLOT_DPI)
    plt.close(figure)

    return output_path

def save_reliability_plots(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    _validate_metrics(metrics_df)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for degradation in _get_degradations(metrics_df):
        condition_df = _condition_frame(metrics_df, degradation)
        saved_paths.append(
            _save_accuracy_confidence_plot(condition_df, degradation, output_dir)
        )

    metric_plots = [
        ("ece", "Expected calibration error", "ece_by_severity.png"),
        (
            "confidence_accuracy_gap",
            "Confidence-accuracy gap",
            "confidence_accuracy_gap_by_severity.png"
        ),
        ("hcer", "High-confidence error rate", "hcer_by_severity.png")
    ]

    for metric, ylabel, filename in metric_plots:
        saved_paths.append(
            _save_metric_plot(metrics_df, metric, ylabel, filename, output_dir)
        )

    return saved_paths