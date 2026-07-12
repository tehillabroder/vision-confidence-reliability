"""Create trust_signal.json from a metrics_summary.csv file."""

import argparse
import json
from pathlib import Path
import pandas as pd
from src.evaluation.trust_signal import assign_trust_signal

def main():
    parser = argparse.ArgumentParser(
        description="Create trust signals from experiment metrics"
    )
    parser.add_argument("--metrics", required=True, help="Path to metrics_summary.csv")
    parser.add_argument("--output", required=True, help="Path to trust_signal.json")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.read_csv(metrics_path)

    # use the single severity 0 row as the undegraded comparison baseline
    baseline_rows = metrics_df[
        (metrics_df["degradation"] == "none")
        & (metrics_df["severity"] == 0)
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one undegraded baseline row.")

    baseline_metrics = baseline_rows.iloc[0].to_dict()
    trust_records = []

    # compare every evaluation condition against the same baseline
    for _, row in metrics_df.iterrows():
        condition_metrics = row.to_dict()
        trust_records.append(
            assign_trust_signal(
                condition_metrics=condition_metrics,
                baseline_metrics=baseline_metrics,
            )
        )

    # indent makes the saved results easier to inspect
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(trust_records, file, indent=2)

    print(f"Saved trust signals to {output_path}")

if __name__ == "__main__":
    main()