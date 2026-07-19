"""Create trust signals from saved experiment metrics."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.trust_signal import assign_trust_signal
from src.utils.config import load_config
REQUIRED_METRIC_COLUMNS = {
    "dataset",
    "model",
    "degradation",
    "severity",
    "accuracy",
    "ece",
    "confidence_accuracy_gap",
    "hcer_fixed",
    "hcer_adaptive"
}

def build_trust_records(metrics_df: pd.DataFrame, trust_policy: dict) -> list[dict]:
    missing_columns = REQUIRED_METRIC_COLUMNS - set(metrics_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Metrics summary is missing columns: {missing}.")

    # isolate the single uncorrupted control row to serve as a uniform baseline for all evaluations
    baseline_rows = metrics_df[
        (metrics_df["degradation"] == "none")
        & (metrics_df["severity"] == 0)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one undegraded baseline row.")

    baseline_metrics = baseline_rows.iloc[0].to_dict()
    # measure each degraded scenario against the exact same stable baseline to map relative error growth
    return [
        assign_trust_signal(row.to_dict(), baseline_metrics, trust_policy)
        for _, row in metrics_df.iterrows()
    ]

def main() -> None:
    parser = argparse.ArgumentParser(description="Create trust signals from experiment metrics")
    parser.add_argument("--config", default="configs/mnist.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["output_dir"])
    metrics_path = output_dir / "metrics_summary.csv"
    output_path = output_dir / "trust_signal.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    trust_records = build_trust_records(metrics_df, config["trust_policy"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # indentation keeps the warning evidence easy to inspect
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(trust_records, output_file, indent=2)

    print(f"Saved {len(trust_records)} trust signals to {output_path}")

if __name__ == "__main__":
    main()