"""Create final GTSRB evidence figures and table."""

import argparse
from pathlib import Path
import pandas as pd
from src.reporting.final_evidence_plots import save_final_evidence_outputs

def main() -> None:
    parser = argparse.ArgumentParser(description="Create final figures from saved GTSRB evidence")
    parser.add_argument("--evidence-dir", default="results/gtsrb_evidence_analysis")
    parser.add_argument("--output-dir", default="results/gtsrb_evidence_analysis/final_outputs")
    parser.add_argument("--overwrite", action="store_true", help="Allow existing final evidence outputs to be replaced.")
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    paired_failures = pd.read_csv(evidence_dir / "paired_model_failures.csv")
    confidence_diagnostics = pd.read_csv(evidence_dir / "confidence_diagnostics.csv")
    trust_attribution = pd.read_csv(evidence_dir / "trust_rule_attribution.csv")
    confidence_transitions = pd.read_csv(evidence_dir / "prediction_confidence_transitions.csv")

    saved_paths = save_final_evidence_outputs(paired_failures, confidence_diagnostics, trust_attribution, confidence_transitions, Path(args.output_dir), overwrite=args.overwrite)

    for path in saved_paths:
        print(f"Saved {path}")

if __name__ == "__main__":
    main()