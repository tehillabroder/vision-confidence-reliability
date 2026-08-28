"""Build GTSRB evidence tables from saved evaluation outputs."""

import argparse
from pathlib import Path
import pandas as pd
from src.reporting.evidence_analysis import (
    build_class_failure_summary, build_confidence_diagnostics, build_paired_model_failures,
    build_prediction_confidence_transitions, build_trust_rule_ablation, build_trust_rule_attribution,
    load_trust_records
)
from src.utils.config import load_config
from src.utils.outputs import check_output_paths

OUTPUT_FILENAMES = {
    "paired_model_failures": "paired_model_failures.csv",
    "class_failure_summary": "class_failure_summary.csv",
    "confidence_diagnostics": "confidence_diagnostics.csv",
    "trust_rule_attribution": "trust_rule_attribution.csv",
    "trust_rule_ablation": "trust_rule_ablation.csv",
    "prediction_confidence_transitions": "prediction_confidence_transitions.csv"
}

def _load_evidence(config_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[dict]]:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    metrics_path = output_dir / "metrics_summary.csv"
    predictions_path = output_dir / "predictions.csv"
    trust_path = output_dir / "trust_signal.json"

    for path in (metrics_path, predictions_path, trust_path):
        if not path.exists():
            raise FileNotFoundError(f"Saved evidence not found: {path}")

    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    trust_records = load_trust_records(trust_path)

    if metrics["model"].drop_duplicates().tolist() != [config["model"]]:
        raise ValueError(f"Metrics summary does not match configured model {config['model']}.")
    if predictions["model"].drop_duplicates().tolist() != [config["model"]]:
        raise ValueError(f"Predictions do not match configured model {config['model']}.")

    return config, metrics, predictions, trust_records

def build_evidence_tables(
    baseline_config: dict,
    baseline_metrics: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    baseline_trust: list[dict],
    stronger_config: dict,
    stronger_metrics: pd.DataFrame,
    stronger_predictions: pd.DataFrame,
    stronger_trust: list[dict]
) -> dict[str, pd.DataFrame]:
    if baseline_config["dataset"] != "GTSRB" or stronger_config["dataset"] != "GTSRB":
        raise ValueError("Evidence analysis requires two GTSRB evaluations.")
    if baseline_config["trust_policy"] != stronger_config["trust_policy"]:
        raise ValueError("Evidence analysis requires matching trust policies.")

    # these are paired-model findings, so both runs need the same evaluation setup
    paired_settings = (
        "seed",
        "validation_split",
        "validation_track_hash",
        "track_overlap",
        "fixed_hcer_threshold",
        "rank_hcer_top_fraction",
        "ece_bins"
    )
    for column in paired_settings:
        baseline_values = baseline_metrics[column].drop_duplicates().tolist()
        stronger_values = stronger_metrics[column].drop_duplicates().tolist()
        if len(baseline_values) != 1 or baseline_values != stronger_values:
            raise ValueError(f"Evidence analysis requires matching {column}.")

    confidence_diagnostics = pd.concat([
        build_confidence_diagnostics(baseline_predictions, baseline_metrics),
        build_confidence_diagnostics(stronger_predictions, stronger_metrics)
    ], ignore_index=True)

    trust_rule_attribution = pd.concat([
        build_trust_rule_attribution(baseline_metrics, baseline_config["trust_policy"], baseline_trust),
        build_trust_rule_attribution(stronger_metrics, stronger_config["trust_policy"], stronger_trust)
    ], ignore_index=True)

    trust_rule_ablation = pd.concat([
        build_trust_rule_ablation(baseline_metrics, baseline_config["trust_policy"]),
        build_trust_rule_ablation(stronger_metrics, stronger_config["trust_policy"])
    ], ignore_index=True)

    prediction_confidence_transitions = pd.concat([
        build_prediction_confidence_transitions(baseline_predictions),
        build_prediction_confidence_transitions(stronger_predictions)
    ], ignore_index=True)

    tables = {
        "paired_model_failures": build_paired_model_failures(baseline_predictions, stronger_predictions),
        "class_failure_summary": build_class_failure_summary(baseline_predictions, stronger_predictions),
        "confidence_diagnostics": confidence_diagnostics,
        "trust_rule_attribution": trust_rule_attribution,
        "trust_rule_ablation": trust_rule_ablation,
        "prediction_confidence_transitions": prediction_confidence_transitions
    }
    provenance = {
        "validation_split": baseline_metrics.iloc[0]["validation_split"],
        "validation_track_hash": baseline_metrics.iloc[0]["validation_track_hash"],
        "track_overlap": int(baseline_metrics.iloc[0]["track_overlap"])
    }
    # carry the split fingerprint into every table so the findings can be traced back later
    for table in tables.values():
        for column, value in provenance.items():
            table[column] = value

    return tables

def main() -> None:
    parser = argparse.ArgumentParser(description="Build deeper evidence tables from completed GTSRB evaluations")
    parser.add_argument("--baseline-config", default="configs/gtsrb.yaml")
    parser.add_argument("--stronger-config", default="configs/gtsrb_resnet18.yaml")
    parser.add_argument("--output-dir", default="results/gtsrb_evidence_analysis")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing evidence-analysis tables to be replaced.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    check_output_paths(output_paths.values(), overwrite=args.overwrite)

    baseline = _load_evidence(Path(args.baseline_config))
    stronger = _load_evidence(Path(args.stronger_config))
    tables = build_evidence_tables(*baseline, *stronger)

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(output_paths[name], index=False)
        print(f"Saved {len(table)} rows to {output_paths[name]}")

if __name__ == "__main__":
    main()