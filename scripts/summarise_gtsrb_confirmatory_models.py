"""Build the limited three-model GTSRB confirmatory summary."""

import argparse
from pathlib import Path
import pandas as pd
from src.reporting.evidence_analysis import load_trust_records
from src.reporting.model_comparison import build_confirmatory_model_summary
from src.utils.config import load_config
from src.utils.outputs import check_output_paths

OUTPUT_FILENAME = "confirmatory_model_summary.csv"

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the confirmatory GTSRB model summary")
    parser.add_argument("--baseline-config", default="configs/gtsrb.yaml")
    parser.add_argument("--main-config", default="configs/gtsrb_resnet18.yaml")
    parser.add_argument("--confirmatory-config", default="configs/gtsrb_mobilenet_v2.yaml")
    parser.add_argument("--output-dir", default="results/gtsrb_model_comparison")
    parser.add_argument("--overwrite", action="store_true", help="Allow the existing confirmatory table to be replaced.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_path = output_dir / OUTPUT_FILENAME
    check_output_paths([output_path], overwrite=args.overwrite)

    baseline = _load_evidence(Path(args.baseline_config))
    main_model = _load_evidence(Path(args.main_config))
    confirmatory = _load_evidence(Path(args.confirmatory_config))
    runs = [baseline, main_model, confirmatory]

    if any(run[0]["dataset"] != "GTSRB" for run in runs):
        raise ValueError("Confirmatory comparison requires GTSRB evaluations.")

    trust_policy = baseline[0]["trust_policy"]
    if any(run[0]["trust_policy"] != trust_policy for run in runs[1:]):
        raise ValueError("Confirmatory comparison requires matching trust policies.")

    table = build_confirmatory_model_summary([
        (metrics, predictions, trust)
        for _, metrics, predictions, trust in runs
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Saved {len(table)} rows to {output_path}")

if __name__ == "__main__":
    main()