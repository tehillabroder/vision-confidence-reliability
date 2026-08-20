"""Compare completed GTSRB model evaluations."""

import argparse
import json
from pathlib import Path
import pandas as pd
from src.reporting.model_comparison import build_model_comparison, build_trust_transition_comparison, save_model_comparison_plots
from src.utils.config import load_config

def _load_trust_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Trust signal not found: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("Trust signal must contain a non-empty list.")

    return records

def _load_evidence(config_path: Path) -> tuple[dict, pd.DataFrame, list[dict]]:
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    metrics_path = output_dir / "metrics_summary.csv"
    trust_path = output_dir / "trust_signal.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {metrics_path}")

    metrics_df = pd.read_csv(metrics_path)
    trust_records = _load_trust_records(trust_path)
    metric_models = metrics_df["model"].drop_duplicates().tolist()
    trust_models = sorted({record.get("model") for record in trust_records})

    # make sure a config does not accidentally load a different model's output folder
    if metric_models != [config["model"]] or trust_models != [config["model"]]:
        raise ValueError(f"Saved evidence does not match configured model {config['model']}.")

    return config, metrics_df, trust_records

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed GTSRB model evaluations")
    parser.add_argument("--baseline-config", default="configs/gtsrb.yaml")
    parser.add_argument("--stronger-config", default="configs/gtsrb_resnet18.yaml")
    parser.add_argument("--output-dir", default="results/gtsrb_model_comparison")
    args = parser.parse_args()

    # pull in both configs and their saved runs
    baseline_config, baseline_metrics, baseline_trust = _load_evidence(Path(args.baseline_config))
    stronger_config, stronger_metrics, stronger_trust = _load_evidence(Path(args.stronger_config))

    # keep it strictly GTSRB to GTSRB for now
    if baseline_config["dataset"] != "GTSRB" or stronger_config["dataset"] != "GTSRB":
        raise ValueError("Model comparison requires two GTSRB evaluations.")

    comparison = build_model_comparison(baseline_metrics, stronger_metrics)
    trust_comparison = build_trust_transition_comparison(baseline_trust, stronger_trust)
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "model_comparison.csv"
    trust_path = output_dir / "trust_transition_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    trust_comparison.to_csv(trust_path, index=False)
    plot_paths = save_model_comparison_plots(baseline_metrics, stronger_metrics, plots_dir)

    print(f"Compared {baseline_config['model']} with {stronger_config['model']}")
    print(f"Saved model comparison to {comparison_path}")
    print(f"Saved trust transitions to {trust_path}")
    print(f"Saved {len(plot_paths)} comparison plots to {plots_dir}")

if __name__ == "__main__":
    main()