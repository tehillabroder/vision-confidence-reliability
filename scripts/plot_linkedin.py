"""Create a LinkedIn-ready confidence-accuracy gap plot."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

INPUT_PATH = Path("results/mnist_degradation_eval/metrics_summary.csv")
OUTPUT_PATH = Path("results/linkedin/confidence_accuracy_gap.png")

DEGRADATIONS = ["blur", "noise", "low_light"]

DISPLAY_NAMES = {
    "blur": "Blur",
    "noise": "Gaussian noise",
    "low_light": "Low light"
}

PALETTE = {
    "blur": "#91BDF5",
    "noise": "#F4AFC1",
    "low_light": "#A9D9CA"
}

BACKGROUND = "#FCFCFE"
GRID = "#E9EBF2"
SPINE = "#D8DCE7"
TEXT = "#303442"
MUTED_TEXT = "#6B7283"
ZERO_LINE = "#9BA4B4"
ANNOTATION_BACKGROUND = "#FFF3F7"

def load_metrics() -> pd.DataFrame:
    metrics_df = pd.read_csv(INPUT_PATH)
    required_columns = {"degradation", "severity", "confidence_accuracy_gap"}
    missing_columns = required_columns - set(metrics_df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    return metrics_df

def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.grid(axis="y", color=GRID, linewidth=1)
    axis.set_axisbelow(True)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.spines["left"].set_linewidth(1)
    axis.spines["bottom"].set_linewidth(1)

    axis.tick_params(
        axis="both",
        color=TEXT,
        labelcolor=TEXT,
        labelsize=11,
        pad=7
    )

def plot_degradation_lines(axis: plt.Axes, metrics_df: pd.DataFrame) -> None:
    for degradation in DEGRADATIONS:
        condition_df = metrics_df[
            metrics_df["degradation"] == degradation
        ].sort_values("severity")

        axis.plot(
            condition_df["severity"],
            condition_df["confidence_accuracy_gap"],
            color=PALETTE[degradation],
            marker="o",
            linewidth=3,
            markersize=8,
            markeredgewidth=1.5,
            markeredgecolor=BACKGROUND,
            solid_capstyle="round",
            label=DISPLAY_NAMES[degradation]
        )

def annotate_noise_flip(axis: plt.Axes, metrics_df: pd.DataFrame) -> None:
    noise_point = metrics_df[
        (metrics_df["degradation"] == "noise")
        & (metrics_df["severity"] == 5)
    ]

    if noise_point.empty:
        return

    x_value = noise_point["severity"].iloc[0]
    y_value = noise_point["confidence_accuracy_gap"].iloc[0]

    axis.annotate(
        "Noise becomes overconfident",
        xy=(x_value, y_value),
        xytext=(4.02, 0.086),
        ha="left",
        va="center",
        fontsize=10.5,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": ANNOTATION_BACKGROUND,
            "edgecolor": "none"
        },
        arrowprops={
            "arrowstyle": "->",
            "color": ZERO_LINE,
            "linewidth": 1.2,
            "connectionstyle": "arc3,rad=0.08"
        }
    )

def add_figure_text(figure: plt.Figure) -> None:
    figure.text(
        0.105,
        0.945,
        "Different degradations affect confidence in different ways",
        fontsize=18,
        color=TEXT,
        fontweight="semibold",
        ha="left"
    )

    figure.text(
        0.105,
        0.895,
        "MNIST classifier under blur, Gaussian noise and low light",
        fontsize=11.5,
        color=MUTED_TEXT,
        ha="left"
    )

def save_linkedin_plot(metrics_df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, 7.2), facecolor=BACKGROUND)

    # reserve separate space for headings, legend and chart
    figure.subplots_adjust(
        left=0.105,
        right=0.955,
        bottom=0.14,
        top=0.75
    )

    style_axis(axis)
    plot_degradation_lines(axis, metrics_df)

    axis.axhline(
        0,
        color=ZERO_LINE,
        linewidth=1.4,
        linestyle=(0, (4, 3))
    )

    annotate_noise_flip(axis, metrics_df)
    add_figure_text(figure)

    axis.set_xlabel(
        "Degradation severity",
        fontsize=12,
        color=TEXT,
        labelpad=14
    )

    axis.set_ylabel(
        "Confidence minus accuracy",
        fontsize=12,
        color=TEXT,
        labelpad=14
    )

    axis.set_xticks(range(1, 6))
    axis.set_xlim(0.85, 5.15)
    axis.set_ylim(-0.56, 0.12)

    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.025),
        ncol=3,
        frameon=False,
        fontsize=11,
        handlelength=2.2,
        columnspacing=2.2,
        borderaxespad=0
    )

    axis.text(
        0.99,
        0.84,
        "Overconfident",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED_TEXT
    )

    axis.text(
        0.99,
        0.035,
        "Underconfident",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=MUTED_TEXT
    )

    figure.savefig(
        OUTPUT_PATH,
        dpi=300,
        facecolor=BACKGROUND
    )

    plt.close(figure)

def main() -> None:
    metrics_df = load_metrics()
    save_linkedin_plot(metrics_df)
    print(f"Saved LinkedIn plot to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()