"""Create plots from a saved metrics summary."""

import argparse
from pathlib import Path
import pandas as pd

from src.reporting.plots import save_reliability_plots

def main():
    parser = argparse.ArgumentParser(description="Create reliability plots from experiment metrics")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics_summary.csv"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder for saved plots"
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    output_dir = Path(args.output_dir)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    saved_paths = save_reliability_plots(metrics_df, output_dir)

    for path in saved_paths:
        print(f"Saved {path}")

if __name__ == "__main__":
    main()
