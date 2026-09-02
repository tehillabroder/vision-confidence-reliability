"""Build bootstrap uncertainty estimates from saved GTSRB evidence."""

import argparse
from pathlib import Path
import pandas as pd
from src.reporting.evidence_analysis import build_bootstrap_uncertainty
from src.utils.config import load_config
from src.utils.outputs import check_output_paths

OUTPUT_FILENAME = "bootstrap_uncertainty.csv"
# reject incomplete prediction evidence
EXPECTED_TEST_IMAGES = 12_630
PAIRED_SETTINGS = (
    "seed",
    "validation_split",
    "validation_track_hash",
    "track_overlap"
)

def _load_saved_predictions(config_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    metrics_path = output_dir / "metrics_summary.csv"
    predictions_path = output_dir / "predictions.csv"

    for path in (metrics_path, predictions_path):
        if not path.exists():
            raise FileNotFoundError(f"Saved evidence not found: {path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)

    if metrics["model"].drop_duplicates().tolist() != [config["model"]]:
        raise ValueError(f"Metrics summary does not match configured model {config['model']}.")
    if predictions["model"].drop_duplicates().tolist() != [config["model"]]:
        raise ValueError(f"Predictions do not match configured model {config['model']}.")

    return config, metrics, predictions

def _matching_provenance(
    baseline_config: dict,
    baseline_metrics: pd.DataFrame,
    stronger_config: dict,
    stronger_metrics: pd.DataFrame
) -> dict:
    if baseline_config["dataset"] != "GTSRB" or stronger_config["dataset"] != "GTSRB":
        raise ValueError("Bootstrap analysis requires two GTSRB evaluations.")
    if baseline_config["model"] != "GTSRBCNN" or stronger_config["model"] != "ResNet18":
        raise ValueError(
            "Bootstrap analysis requires GTSRBCNN as the baseline "
            "and ResNet18 as the stronger model."
        )

    for column in PAIRED_SETTINGS:
        baseline_values = baseline_metrics[column].drop_duplicates().tolist()
        stronger_values = stronger_metrics[column].drop_duplicates().tolist()
        if len(baseline_values) != 1 or baseline_values != stronger_values:
            raise ValueError(f"Bootstrap analysis requires matching {column}.")

    return {
        "validation_split": baseline_metrics.iloc[0]["validation_split"],
        "validation_track_hash": baseline_metrics.iloc[0]["validation_track_hash"],
        "track_overlap": int(baseline_metrics.iloc[0]["track_overlap"])
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build selected bootstrap intervals from saved GTSRB predictions"
    )
    parser.add_argument("--baseline-config", default="configs/gtsrb.yaml")
    parser.add_argument("--stronger-config", default="configs/gtsrb_resnet18.yaml")
    parser.add_argument("--output-dir", default="results/gtsrb_evidence_analysis")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow the existing bootstrap table to be replaced."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / OUTPUT_FILENAME
    check_output_paths([output_path], overwrite=args.overwrite)

    baseline_config, baseline_metrics, baseline_predictions = _load_saved_predictions(
        Path(args.baseline_config)
    )
    stronger_config, stronger_metrics, stronger_predictions = _load_saved_predictions(
        Path(args.stronger_config)
    )
    provenance = _matching_provenance(
        baseline_config,
        baseline_metrics,
        stronger_config,
        stronger_metrics
    )
    table = build_bootstrap_uncertainty(
        baseline_predictions,
        stronger_predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed
    )

    comparison_rows = table[
        table["analysis"].isin(
            ("accuracy_difference", "failure_detection_auroc_difference")
        )
    ]
    if not comparison_rows["num_images"].eq(EXPECTED_TEST_IMAGES).all():
        raise ValueError(
            "Bootstrap analysis expected 12,630 paired predictions "
            "at noise severity 5."
        )

    for column, value in provenance.items():
        table[column] = value

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved {len(table)} rows to {output_path}")

if __name__ == "__main__":
    main()