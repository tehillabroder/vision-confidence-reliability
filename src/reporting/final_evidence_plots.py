"""Create final figures from saved GTSRB evidence."""

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import pandas as pd
from src.utils.outputs import check_output_paths

plt.switch_backend("Agg")

PLOT_DPI = 200
PASTEL_BLUE = "#d0e4f8"
PASTEL_GREEN = "#ddf1ea"
PASTEL_PINK = "#ffdbe6"
PASTEL_GREY = "#ddd5dc"
PROVENANCE_COLUMNS = ["validation_split", "validation_track_hash", "track_overlap"]
DEGRADATIONS = ["blur", "noise", "low_light"]
SIGNAL_VALUES = {"trust": 0, "caution": 1, "do_not_trust": 2}
SIGNAL_TEXT = {0: "T", 1: "C", 2: "DNT"}
FINAL_OUTPUT_FILENAMES = [
    "warning_timing_and_attribution.png",
    "severe_noise_failure_profiles.png",
    "fixed_hcer_context.png",
    "confidence_changes_on_failures.png",
    "severe_noise_results.csv"
]

def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}.")

def _provenance(frame: pd.DataFrame, name: str) -> tuple:
    _require_columns(frame, set(PROVENANCE_COLUMNS), name)
    values = frame[PROVENANCE_COLUMNS].drop_duplicates()

    if len(values) != 1:
        raise ValueError(f"{name} must contain one validation split fingerprint.")

    return tuple(values.iloc[0].tolist())

def _get_models(paired_failures: pd.DataFrame) -> tuple[str, str]:
    pairs = paired_failures[["baseline_model", "stronger_model"]].drop_duplicates()

    if len(pairs) != 1:
        raise ValueError("Paired failures must contain one baseline and stronger model pair.")

    baseline_model = str(pairs.iloc[0]["baseline_model"])
    stronger_model = str(pairs.iloc[0]["stronger_model"])
    return baseline_model, stronger_model

def _validate_evidence(paired_failures: pd.DataFrame, confidence_diagnostics: pd.DataFrame, trust_attribution: pd.DataFrame, confidence_transitions: pd.DataFrame) -> None:
    paired_required = {"dataset", "baseline_model", "stronger_model", "degradation", "severity", "both_correct", "baseline_only_correct", "stronger_only_correct", "both_wrong", "both_wrong_same_class_count", "same_wrong_class_rate_among_both_wrong"}
    diagnostic_required = {"dataset", "model", "degradation", "severity", "accuracy", "mean_confidence", "confidence_accuracy_gap", "hcer_fixed", "hcer_fixed_coverage", "hcer_fixed_conditional_error", "failure_detection_auroc", "rank_hcer_top_fraction", "rank_high_confidence_accuracy"}
    trust_required = {"dataset", "model", "degradation", "severity", "performance_signal", "confidence_signal", "trust_signal"}
    transition_required = {"dataset", "model", "degradation", "from_severity", "to_severity", "wrong_at_both_mean_confidence_change", "wrong_at_both_confidence_increased_rate", "correct_to_wrong_mean_confidence_change"}

    _require_columns(paired_failures, paired_required, "Paired failures")
    _require_columns(confidence_diagnostics, diagnostic_required, "Confidence diagnostics")
    _require_columns(trust_attribution, trust_required, "Trust attribution")
    _require_columns(confidence_transitions, transition_required, "Confidence transitions")

    frames = [
        ("Paired failures", paired_failures),
        ("Confidence diagnostics", confidence_diagnostics),
        ("Trust attribution", trust_attribution),
        ("Confidence transitions", confidence_transitions)
    ]
    fingerprints = [_provenance(frame, name) for name, frame in frames]

    if len(set(fingerprints)) != 1:
        raise ValueError("Final evidence plots require matching validation split fingerprints.")

    baseline_model, stronger_model = _get_models(paired_failures)
    expected_models = {baseline_model, stronger_model}

    for name, frame in frames:
        if frame["dataset"].drop_duplicates().tolist() != ["GTSRB"]:
            raise ValueError(f"{name} must contain GTSRB evidence only.")

    for name, frame in frames[1:]:
        if set(frame["model"].drop_duplicates()) != expected_models:
            raise ValueError(f"{name} must contain the paired GTSRB models.")

def build_severe_noise_results_table(paired_failures: pd.DataFrame, confidence_diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Build the compact severe-noise comparison table."""
    baseline_model, stronger_model = _get_models(paired_failures)
    severe = confidence_diagnostics[(confidence_diagnostics["degradation"] == "noise") & (confidence_diagnostics["severity"] == 5)].copy()

    if set(severe["model"]) != {baseline_model, stronger_model} or len(severe) != 2:
        raise ValueError("Expected one Gaussian noise severity 5 row for each paired model.")
    if not severe["rank_hcer_top_fraction"].eq(0.10).all():
        raise ValueError("Severe-noise table expects the saved rank diagnostic to use the top 10%.")

    columns = [
        "accuracy", "mean_confidence", "confidence_accuracy_gap", "hcer_fixed",
        "hcer_fixed_coverage", "hcer_fixed_conditional_error", "failure_detection_auroc",
        "rank_high_confidence_accuracy"
    ]
    result = severe.set_index("model").loc[[baseline_model, stronger_model], columns].reset_index()
    return result.rename(columns={"rank_high_confidence_accuracy": "top_10_confidence_accuracy"})

def _warning_signal_frame(trust_attribution: pd.DataFrame) -> pd.DataFrame:
    columns = ["model", "degradation", "severity", "performance_signal", "confidence_signal", "trust_signal"]
    result = trust_attribution[columns].drop_duplicates()

    if result.duplicated(["model", "degradation", "severity"]).any():
        raise ValueError("Trust attribution contains inconsistent signals for the same condition.")

    return result

def _save_warning_timing_plot(trust_attribution: pd.DataFrame, models: tuple[str, str], output_path: Path) -> None:
    signal_frame = _warning_signal_frame(trust_attribution)
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 5.4), sharey=True)
    colour_map = ListedColormap([PASTEL_GREEN, PASTEL_BLUE, PASTEL_PINK])
    signal_columns = [("performance_signal", "Performance"), ("confidence_signal", "Confidence"), ("trust_signal", "Overall")]

    for axis, degradation in zip(axes, DEGRADATIONS):
        rows = []
        row_labels = []

        for model in models:
            for column, label in signal_columns:
                values = []

                # severity zero is the reference condition, so show only degraded conditions
                for severity in range(1, 6):
                    condition = signal_frame[(signal_frame["model"] == model) & (signal_frame["degradation"] == degradation) & (signal_frame["severity"] == severity)]
                    if len(condition) != 1:
                        raise ValueError(f"Expected one {model} {degradation} severity {severity} trust condition.")
                    values.append(SIGNAL_VALUES[condition.iloc[0][column]])

                rows.append(values)
                row_labels.append(f"{model} {label}")

        axis.imshow(rows, cmap=colour_map, vmin=0, vmax=2, aspect="auto")
        axis.set_title(degradation.replace("_", " ").title())
        axis.set_xlabel("Severity")
        axis.set_xticks(range(5), labels=range(1, 6))
        axis.set_yticks(range(len(row_labels)), labels=row_labels)

        if axis is not axes[0]:
            axis.tick_params(labelleft=False)

        for row_index, values in enumerate(rows):
            for column_index, value in enumerate(values):
                axis.text(column_index, row_index, SIGNAL_TEXT[value], ha="center", va="center", fontsize=8)

    legend = [
        Patch(facecolor=PASTEL_GREEN, edgecolor="#666666", label="Trust"),
        Patch(facecolor=PASTEL_BLUE, edgecolor="#666666", label="Caution"),
        Patch(facecolor=PASTEL_PINK, edgecolor="#666666", label="Do Not Trust")
    ]
    figure.suptitle("GTSRB Warning Timing and Attribution")
    figure.legend(handles=legend, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

def _save_severe_noise_plot(paired_failures: pd.DataFrame, confidence_diagnostics: pd.DataFrame, models: tuple[str, str], output_path: Path) -> None:
    paired = paired_failures[(paired_failures["degradation"] == "noise") & (paired_failures["severity"] == 5)]

    if len(paired) != 1:
        raise ValueError("Expected one paired Gaussian noise severity 5 row.")

    paired = paired.iloc[0]
    table = build_severe_noise_results_table(paired_failures, confidence_diagnostics).set_index("model")
    baseline_model, stronger_model = models
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))

    outcome_labels = ["Both\nCorrect", f"{baseline_model}\nOnly", f"{stronger_model}\nOnly", "Both\nWrong"]
    outcome_values = [paired["both_correct"], paired["baseline_only_correct"], paired["stronger_only_correct"], paired["both_wrong"]]
    bars = axes[0].bar(range(4), outcome_values, color=[PASTEL_GREEN, PASTEL_BLUE, PASTEL_PINK, PASTEL_GREY], edgecolor="#666666")
    axes[0].set_title("Paired Prediction Outcomes")
    axes[0].set_ylabel("Images")
    axes[0].set_xticks(range(4), labels=outcome_labels)
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, outcome_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(value):,}", ha="center", va="bottom", fontsize=9)

    same_wrong_count = int(paired["both_wrong_same_class_count"])
    same_wrong_rate = float(paired["same_wrong_class_rate_among_both_wrong"])
    axes[0].text(0.5, -0.22, f"Shared failures with the same wrong class: {same_wrong_count:,} ({same_wrong_rate:.1%})", transform=axes[0].transAxes, ha="center", fontsize=9)

    metric_columns = ["accuracy", "mean_confidence", "failure_detection_auroc", "top_10_confidence_accuracy"]
    metric_labels = ["Accuracy", "Mean\nConfidence", "Failure\nDetection\nAUROC", "Top 10%\nConfidence\nAccuracy"]
    positions = list(range(len(metric_columns)))
    width = 0.36
    baseline_values = [table.loc[baseline_model, column] for column in metric_columns]
    stronger_values = [table.loc[stronger_model, column] for column in metric_columns]

    axes[1].bar([position - width / 2 for position in positions], baseline_values, width=width, color=PASTEL_BLUE, edgecolor="#666666", label=baseline_model)
    axes[1].bar([position + width / 2 for position in positions], stronger_values, width=width, color=PASTEL_PINK, edgecolor="#666666", label=stronger_model)
    axes[1].set_title("Summary and Confidence Diagnostics")
    axes[1].set_ylabel("Score / Rate")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(positions, labels=metric_labels)
    axes[1].tick_params(axis="x", labelsize=8.5)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle("GTSRB Gaussian Noise Severity 5")
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

def _noise_with_baseline(confidence_diagnostics: pd.DataFrame, model: str) -> pd.DataFrame:
    baseline = confidence_diagnostics[(confidence_diagnostics["model"] == model) & (confidence_diagnostics["degradation"] == "none") & (confidence_diagnostics["severity"] == 0)].copy()
    noise = confidence_diagnostics[(confidence_diagnostics["model"] == model) & (confidence_diagnostics["degradation"] == "noise")].copy()

    if len(baseline) != 1 or sorted(noise["severity"].tolist()) != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected clean and Gaussian noise severities 1 to 5 for {model}.")

    baseline["degradation"] = "noise"
    return pd.concat([baseline, noise], ignore_index=True).sort_values("severity")

def _save_hcer_context_plot(confidence_diagnostics: pd.DataFrame, baseline_model: str, output_path: Path) -> None:
    noise = _noise_with_baseline(confidence_diagnostics, baseline_model)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    axes[0].plot(noise["severity"], noise["hcer_fixed"], marker="o", linewidth=2.5, color=PASTEL_BLUE, markeredgecolor="#666666")
    axes[0].set_title("Fixed HCER")
    axes[0].set_xlabel("Gaussian noise severity")
    axes[0].set_ylabel("Error rate")
    axes[0].set_xticks(range(6))
    axes[0].set_ylim(0, max(0.25, float(noise["hcer_fixed"].max()) * 1.2))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(alpha=0.25)

    axes[1].plot(noise["severity"], noise["hcer_fixed_coverage"], marker="o", linewidth=2.5, color=PASTEL_GREEN, markeredgecolor="#666666", label="Coverage above 0.90")
    axes[1].plot(noise["severity"], noise["hcer_fixed_conditional_error"], marker="o", linewidth=2.5, color=PASTEL_PINK, markeredgecolor="#666666", label="Error within selected group")
    axes[1].set_title("What Sits Underneath Fixed HCER")
    axes[1].set_xlabel("Gaussian noise severity")
    axes[1].set_ylabel("Rate")
    axes[1].set_xticks(range(6))
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.suptitle(f"{baseline_model}: Fixed HCER Needs Coverage Context")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

def _save_confidence_transition_plot(confidence_transitions: pd.DataFrame, models: tuple[str, str], output_path: Path) -> None:
    baseline_model, stronger_model = models
    colours = {baseline_model: PASTEL_BLUE, stronger_model: PASTEL_PINK}

    # separate persistent errors from new failures because their changes occur on different scales
    figure, axes = plt.subplots(2, 3, figsize=(13, 8.2), sharex=True, sharey="row")
    row_specs = [
        ("wrong_at_both_mean_confidence_change", "Persistent Errors"),
        ("correct_to_wrong_mean_confidence_change", "New Failures")
    ]

    for column_index, degradation in enumerate(DEGRADATIONS):
        for row_index, (metric_column, row_title) in enumerate(row_specs):
            axis = axes[row_index, column_index]

            for model in models:
                rows = confidence_transitions[(confidence_transitions["model"] == model) & (confidence_transitions["degradation"] == degradation)].sort_values("from_severity")

                if list(zip(rows["from_severity"], rows["to_severity"])) != [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
                    raise ValueError(f"Expected five adjacent {degradation} transitions for {model}.")

                axis.plot(rows["to_severity"], rows[metric_column], marker="o", linewidth=2.5, color=colours[model], markeredgecolor="#666666", label=model)

            axis.axhline(0, linewidth=1, color="#777777")
            axis.set_xticks(range(1, 6), labels=["0→1", "1→2", "2→3", "3→4", "4→5"])
            axis.grid(alpha=0.25)

            if row_index == 0:
                axis.set_title(degradation.replace("_", " ").title())

            if column_index == 0:
                axis.set_ylabel(f"{row_title}\nMean Confidence Change")
                axis.yaxis.set_major_formatter(PercentFormatter(1.0))

            if row_index == 1:
                axis.set_xlabel("Severity Transition")

    # highlight the strongest example without crowding the rest of the figure
    final_transition = confidence_transitions[
        (confidence_transitions["model"] == stronger_model)
        & (confidence_transitions["degradation"] == "noise")
        & (confidence_transitions["from_severity"] == 4)
        & (confidence_transitions["to_severity"] == 5)
    ].iloc[0]

    axes[0, 1].text(
        0.97,
        0.95,
        f"{stronger_model} 4→5\nMean Change: {final_transition['wrong_at_both_mean_confidence_change']:+.1%}\nMore Confident: {final_transition['wrong_at_both_confidence_increased_rate']:.1%}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": PASTEL_GREY, "alpha": 0.9}
    )
    axes[1, 1].text(
        0.97,
        0.95,
        f"{stronger_model} 4→5\nMean Change: {final_transition['correct_to_wrong_mean_confidence_change']:+.1%}",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": PASTEL_GREY, "alpha": 0.9}
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Confidence Changes as the Same Images Become More Degraded", y=0.97)
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    figure.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(figure)

def save_final_evidence_outputs(paired_failures: pd.DataFrame, confidence_diagnostics: pd.DataFrame, trust_attribution: pd.DataFrame, confidence_transitions: pd.DataFrame, output_dir: Path, overwrite: bool = False) -> list[Path]:
    _validate_evidence(paired_failures, confidence_diagnostics, trust_attribution, confidence_transitions)
    output_paths = [output_dir / filename for filename in FINAL_OUTPUT_FILENAMES]
    check_output_paths(output_paths, overwrite=overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = _get_models(paired_failures)

    _save_warning_timing_plot(trust_attribution, models, output_paths[0])
    _save_severe_noise_plot(paired_failures, confidence_diagnostics, models, output_paths[1])
    _save_hcer_context_plot(confidence_diagnostics, models[0], output_paths[2])
    _save_confidence_transition_plot(confidence_transitions, models, output_paths[3])
    build_severe_noise_results_table(paired_failures, confidence_diagnostics).to_csv(output_paths[4], index=False)

    return output_paths