"""Build deeper evidence tables from saved evaluation results."""

from __future__ import annotations
import json
import pandas as pd
from src.evaluation.trust_signal import CONFIDENCE_METRICS, PERFORMANCE_METRICS, RULE_LABELS, assign_trust_signal
from src.metrics.reliability import conditional_high_confidence_error_rate, failure_detection_auroc, high_confidence_coverage

CONDITION_COLUMNS = ["dataset", "seed", "degradation", "severity"]
PREDICTION_PAIR_COLUMNS = CONDITION_COLUMNS + ["image_id"]
PREDICTION_REQUIRED_COLUMNS = set(PREDICTION_PAIR_COLUMNS) | {"model", "true_label", "predicted_label", "correct", "confidence"}
CONFIDENCE_METRIC_COLUMNS = {
    "dataset", "model", "seed", "degradation", "severity", "accuracy",
    "mean_confidence", "confidence_accuracy_gap", "ece", "hcer_fixed",
    "hcer_rank", "hcer_rank_coverage", "rank_hcer_top_fraction",
    "fixed_hcer_threshold", "num_examples"
}
DEGRADATION_ORDER = ("none", "blur", "noise", "low_light")

def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(sorted(missing))}.")

def _single_model(frame: pd.DataFrame, name: str) -> str:
    models = frame["model"].drop_duplicates().tolist()
    if len(models) != 1:
        raise ValueError(f"{name} must contain exactly one model.")
    return str(models[0])

def _sort_conditions(frame: pd.DataFrame, extra_columns: list[str] | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["degradation"] = pd.Categorical(result["degradation"], categories=DEGRADATION_ORDER, ordered=True)
    sort_columns = ["degradation", "severity"]

    if "model" in result.columns:
        sort_columns.insert(0, "model")
    if extra_columns:
        sort_columns.extend(extra_columns)

    return result.sort_values(sort_columns).reset_index(drop=True)

# pair the same images so similar accuracy can't hide different successes and failures
def _pair_predictions(baseline_predictions: pd.DataFrame, stronger_predictions: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    _require_columns(baseline_predictions, PREDICTION_REQUIRED_COLUMNS, "Baseline predictions")
    _require_columns(stronger_predictions, PREDICTION_REQUIRED_COLUMNS, "Stronger-model predictions")
    baseline_model = _single_model(baseline_predictions, "Baseline predictions")
    stronger_model = _single_model(stronger_predictions, "Stronger-model predictions")

    if baseline_model == stronger_model:
        raise ValueError("Paired evidence requires two different models.")

    evidence_columns = PREDICTION_PAIR_COLUMNS + ["true_label", "predicted_label", "correct", "confidence"]
    paired = baseline_predictions[evidence_columns].merge(
        stronger_predictions[evidence_columns],
        on=PREDICTION_PAIR_COLUMNS,
        how="outer",
        suffixes=("_baseline", "_stronger"),
        indicator=True,
        validate="one_to_one"
    )

    if not paired["_merge"].eq("both").all():
        raise ValueError("Prediction evidence does not contain identical image IDs for every condition.")
    if not paired["true_label_baseline"].eq(paired["true_label_stronger"]).all():
        raise ValueError("Prediction evidence contains different true labels for paired images.")

    paired = paired.drop(columns="_merge")
    paired["correct_baseline"] = paired["correct_baseline"].astype(int)
    paired["correct_stronger"] = paired["correct_stronger"].astype(int)
    return _sort_conditions(paired, ["image_id"]), baseline_model, stronger_model

def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)

def build_paired_model_failures(baseline_predictions: pd.DataFrame, stronger_predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired image outcomes for every condition."""
    paired, baseline_model, stronger_model = _pair_predictions(baseline_predictions, stronger_predictions)
    rows = []

    for condition, group in paired.groupby(CONDITION_COLUMNS, sort=False, observed=True):
        dataset, seed, degradation, severity = condition
        baseline_correct = group["correct_baseline"].eq(1)
        stronger_correct = group["correct_stronger"].eq(1)
        both_correct = baseline_correct & stronger_correct
        baseline_only = baseline_correct & ~stronger_correct
        stronger_only = ~baseline_correct & stronger_correct
        both_wrong = ~baseline_correct & ~stronger_correct
        prediction_agreement = group["predicted_label_baseline"].eq(group["predicted_label_stronger"])
        both_wrong_same = both_wrong & prediction_agreement
        num_examples = len(group)
        correct_union = int((baseline_correct | stronger_correct).sum())
        both_wrong_count = int(both_wrong.sum())

        rows.append({
            "dataset": dataset,
            "seed": int(seed),
            "degradation": degradation,
            "severity": int(severity),
            "baseline_model": baseline_model,
            "stronger_model": stronger_model,
            "num_examples": num_examples,
            "baseline_correct_count": int(baseline_correct.sum()),
            "stronger_correct_count": int(stronger_correct.sum()),
            "baseline_accuracy": float(baseline_correct.mean()),
            "stronger_accuracy": float(stronger_correct.mean()),
            "accuracy_delta_stronger_minus_baseline": float(stronger_correct.mean() - baseline_correct.mean()),
            "both_correct": int(both_correct.sum()),
            "baseline_only_correct": int(baseline_only.sum()),
            "stronger_only_correct": int(stronger_only.sum()),
            "both_wrong": both_wrong_count,
            "both_correct_rate": float(both_correct.mean()),
            "baseline_only_correct_rate": float(baseline_only.mean()),
            "stronger_only_correct_rate": float(stronger_only.mean()),
            "both_wrong_rate": float(both_wrong.mean()),
            "correct_set_union_count": correct_union,
            "correct_set_jaccard": _safe_rate(int(both_correct.sum()), correct_union),
            "prediction_agreement_count": int(prediction_agreement.sum()),
            "prediction_agreement_rate": float(prediction_agreement.mean()),
            "both_wrong_same_class_count": int(both_wrong_same.sum()),
            "both_wrong_different_class_count": int((both_wrong & ~prediction_agreement).sum()),
            "same_wrong_class_rate_among_both_wrong": _safe_rate(int(both_wrong_same.sum()), both_wrong_count)
        })

    return _sort_conditions(pd.DataFrame(rows))

def _rank_counts(counts: dict[int, int]) -> dict[int, int]:
    ordered_classes = sorted(counts, key=lambda label: (-counts[label], label))
    return {label: rank for rank, label in enumerate(ordered_classes, start=1)}

def build_class_failure_summary(baseline_predictions: pd.DataFrame, stronger_predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare true-class recall and predicted-class concentration."""
    paired, baseline_model, stronger_model = _pair_predictions(baseline_predictions, stronger_predictions)
    rows = []

    for condition, group in paired.groupby(CONDITION_COLUMNS, sort=False, observed=True):
        dataset, seed, degradation, severity = condition
        labels = sorted(int(label) for label in group["true_label_baseline"].unique())
        baseline_correct_total = int(group["correct_baseline"].sum())
        stronger_correct_total = int(group["correct_stronger"].sum())
        baseline_correct_counts = {}
        stronger_correct_counts = {}
        baseline_prediction_counts = {}
        stronger_prediction_counts = {}

        # look at which true classes survive, and which labels the models start collapsing towards
        for label in labels:
            true_class = group["true_label_baseline"].eq(label)
            baseline_correct_counts[label] = int((true_class & group["correct_baseline"].eq(1)).sum())
            stronger_correct_counts[label] = int((true_class & group["correct_stronger"].eq(1)).sum())
            baseline_prediction_counts[label] = int(group["predicted_label_baseline"].eq(label).sum())
            stronger_prediction_counts[label] = int(group["predicted_label_stronger"].eq(label).sum())

        baseline_correct_ranks = _rank_counts(baseline_correct_counts)
        stronger_correct_ranks = _rank_counts(stronger_correct_counts)
        baseline_prediction_ranks = _rank_counts(baseline_prediction_counts)
        stronger_prediction_ranks = _rank_counts(stronger_prediction_counts)

        for label in labels:
            true_class = group["true_label_baseline"].eq(label)
            true_count = int(true_class.sum())
            baseline_recall = baseline_correct_counts[label] / true_count
            stronger_recall = stronger_correct_counts[label] / true_count

            if baseline_recall > stronger_recall:
                higher_recall_model = baseline_model
            elif stronger_recall > baseline_recall:
                higher_recall_model = stronger_model
            else:
                higher_recall_model = "tie"

            rows.append({
                "dataset": dataset,
                "seed": int(seed),
                "degradation": degradation,
                "severity": int(severity),
                "baseline_model": baseline_model,
                "stronger_model": stronger_model,
                "true_label": label,
                "true_class_count": true_count,
                "baseline_correct_count": baseline_correct_counts[label],
                "stronger_correct_count": stronger_correct_counts[label],
                "baseline_recall": float(baseline_recall),
                "stronger_recall": float(stronger_recall),
                "recall_delta_stronger_minus_baseline": float(stronger_recall - baseline_recall),
                "higher_recall_model": higher_recall_model,
                "baseline_share_of_all_correct": _safe_rate(baseline_correct_counts[label], baseline_correct_total),
                "stronger_share_of_all_correct": _safe_rate(stronger_correct_counts[label], stronger_correct_total),
                "baseline_correct_contribution_rank": baseline_correct_ranks[label],
                "stronger_correct_contribution_rank": stronger_correct_ranks[label],
                "baseline_predicted_count": baseline_prediction_counts[label],
                "stronger_predicted_count": stronger_prediction_counts[label],
                "baseline_prediction_share": float(baseline_prediction_counts[label] / len(group)),
                "stronger_prediction_share": float(stronger_prediction_counts[label] / len(group)),
                "baseline_prediction_rank": baseline_prediction_ranks[label],
                "stronger_prediction_rank": stronger_prediction_ranks[label]
            })

    return _sort_conditions(pd.DataFrame(rows), ["true_label"])

def _validate_metric_conditions(predictions: pd.DataFrame, metrics: pd.DataFrame) -> str:
    _require_columns(predictions, PREDICTION_REQUIRED_COLUMNS, "Predictions")
    _require_columns(metrics, CONFIDENCE_METRIC_COLUMNS, "Metrics summary")
    prediction_model = _single_model(predictions, "Predictions")
    metric_model = _single_model(metrics, "Metrics summary")

    if prediction_model != metric_model:
        raise ValueError("Predictions and metrics summary describe different models.")
    if metrics.duplicated(CONDITION_COLUMNS).any():
        raise ValueError("Metrics summary must contain one row per condition.")

    prediction_conditions = set(map(tuple, predictions[CONDITION_COLUMNS].drop_duplicates().to_numpy()))
    metric_conditions = set(map(tuple, metrics[CONDITION_COLUMNS].to_numpy()))
    if prediction_conditions != metric_conditions:
        raise ValueError("Predictions and metrics summary contain different conditions.")

    return prediction_model

def build_confidence_diagnostics(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Add context that explains what the confidence metrics are hiding."""
    model = _validate_metric_conditions(predictions, metrics)
    metric_lookup = metrics.set_index(CONDITION_COLUMNS)
    rows = []

    for condition, group in predictions.groupby(CONDITION_COLUMNS, sort=False):
        dataset, seed, degradation, severity = condition
        metric = metric_lookup.loc[condition]
        correct = group["correct"].to_numpy()
        confidences = group["confidence"].to_numpy()
        fixed_threshold = float(metric["fixed_hcer_threshold"])

        # HCER can fall when fewer predictions clear the threshold, so keep both parts beside it
        fixed_coverage = high_confidence_coverage(confidences, threshold=fixed_threshold)
        fixed_conditional_error = conditional_high_confidence_error_rate(correct, confidences, threshold=fixed_threshold)

        # AUROC shows whether confidence is still useful for separating correct and wrong predictions
        failure_auroc = failure_detection_auroc(correct, confidences)
        rank_coverage = float(metric["hcer_rank_coverage"])
        rank_hcer = float(metric["hcer_rank"])
        rank_conditional_error = rank_hcer / rank_coverage

        # the raw predictions and saved metrics should still describe the same condition
        if len(group) != int(metric["num_examples"]):
            raise ValueError(f"Prediction count does not match metrics summary for {degradation} severity {severity}.")

        rows.append({
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "degradation": degradation,
            "severity": int(severity),
            "num_examples": len(group),
            "accuracy": float(metric["accuracy"]),
            "mean_confidence": float(metric["mean_confidence"]),
            "confidence_accuracy_gap": float(metric["confidence_accuracy_gap"]),
            "ece": float(metric["ece"]),
            "fixed_hcer_threshold": fixed_threshold,
            "hcer_fixed": float(metric["hcer_fixed"]),
            "hcer_fixed_selected_count": int((confidences >= fixed_threshold).sum()),
            "hcer_fixed_wrong_count": int(((correct == 0) & (confidences >= fixed_threshold)).sum()),
            "hcer_fixed_coverage": fixed_coverage,
            "hcer_fixed_conditional_error": fixed_conditional_error,
            "failure_detection_auroc": failure_auroc,
            "rank_hcer_top_fraction": float(metric["rank_hcer_top_fraction"]),
            "hcer_rank": rank_hcer,
            "hcer_rank_coverage": rank_coverage,
            "hcer_rank_conditional_error": float(rank_conditional_error),
            "rank_high_confidence_accuracy": float(1.0 - rank_conditional_error)
        })

    return _sort_conditions(pd.DataFrame(rows))

def build_prediction_confidence_transitions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Track how confidence changes as the same images become more degraded."""
    _require_columns(predictions, PREDICTION_REQUIRED_COLUMNS, "Predictions")
    model = _single_model(predictions, "Predictions")

    if predictions.duplicated(PREDICTION_PAIR_COLUMNS).any():
        raise ValueError("Predictions must contain one row per image and condition.")

    clean = predictions[(predictions["degradation"] == "none") & (predictions["severity"] == 0)]
    if clean.empty:
        raise ValueError("Predictions must contain the undegraded baseline.")

    rows = []
    degradations = predictions.loc[predictions["degradation"] != "none", "degradation"].drop_duplicates()

    for degradation in degradations:
        degraded = predictions[predictions["degradation"] == degradation]
        severities = sorted(int(value) for value in degraded["severity"].unique())

        if severities != list(range(1, max(severities) + 1)):
            raise ValueError(f"{degradation} severities must be consecutive from 1.")

        transitions = [(0, severities[0])] + list(zip(severities[:-1], severities[1:]))

        for from_severity, to_severity in transitions:
            from_rows = clean if from_severity == 0 else degraded[degraded["severity"] == from_severity]
            to_rows = degraded[degraded["severity"] == to_severity]
            pair_columns = ["dataset", "seed", "image_id", "true_label"]

            # follow the same images one step at a time instead of comparing condition averages
            paired = from_rows[pair_columns + ["correct", "confidence"]].merge(
                to_rows[pair_columns + ["correct", "confidence"]],
                on=pair_columns,
                suffixes=("_from", "_to"),
                validate="one_to_one"
            )

            if len(paired) != len(from_rows) or len(paired) != len(to_rows):
                raise ValueError("Adjacent severities must contain the same images and true labels.")

            from_correct = paired["correct_from"].eq(1)
            to_correct = paired["correct_to"].eq(1)
            confidence_change = paired["confidence_to"] - paired["confidence_from"]
            confidence_increased = confidence_change > 0
            wrong_at_both = ~from_correct & ~to_correct
            correct_to_wrong = from_correct & ~to_correct
            wrong_to_correct = ~from_correct & to_correct

            # persistent errors show whether worsening images can make an already wrong prediction more confident
            wrong_at_both_count = int(wrong_at_both.sum())
            correct_to_wrong_count = int(correct_to_wrong.sum())

            rows.append({
                "dataset": paired.iloc[0]["dataset"],
                "model": model,
                "seed": int(paired.iloc[0]["seed"]),
                "degradation": degradation,
                "from_severity": from_severity,
                "to_severity": to_severity,
                "num_images": len(paired),
                "from_accuracy": float(from_correct.mean()),
                "to_accuracy": float(to_correct.mean()),
                "accuracy_change": float(to_correct.mean() - from_correct.mean()),
                "from_mean_confidence": float(paired["confidence_from"].mean()),
                "to_mean_confidence": float(paired["confidence_to"].mean()),
                "mean_confidence_change": float(confidence_change.mean()),
                "confidence_increased_count": int(confidence_increased.sum()),
                "confidence_increased_rate": float(confidence_increased.mean()),
                "wrong_at_both_count": wrong_at_both_count,
                "wrong_at_both_confidence_increased_count": int((wrong_at_both & confidence_increased).sum()),
                "wrong_at_both_confidence_increased_rate": _safe_rate(
                    int((wrong_at_both & confidence_increased).sum()),
                    wrong_at_both_count
                ),
                "wrong_at_both_mean_confidence_change": (
                    float(confidence_change[wrong_at_both].mean())
                    if wrong_at_both_count else None
                ),
                "correct_to_wrong_count": correct_to_wrong_count,
                "correct_to_wrong_confidence_increased_count": int((correct_to_wrong & confidence_increased).sum()),
                "correct_to_wrong_confidence_increased_rate": _safe_rate(
                    int((correct_to_wrong & confidence_increased).sum()),
                    correct_to_wrong_count
                ),
                "correct_to_wrong_mean_confidence_change": (
                    float(confidence_change[correct_to_wrong].mean())
                    if correct_to_wrong_count else None
                ),
                "wrong_to_correct_count": int(wrong_to_correct.sum())
            })

    result = pd.DataFrame(rows)
    result["degradation"] = pd.Categorical(result["degradation"], categories=DEGRADATION_ORDER, ordered=True)

    return result.sort_values(["model", "degradation", "from_severity"]).reset_index(drop=True)

def _validate_saved_trust(recomputed: dict, saved: dict, condition: tuple) -> None:
    if recomputed["trust_signal"] != saved.get("trust_signal"):
        raise ValueError(f"Saved trust signal does not reconstruct for {condition[0]} severity {condition[1]}.")
    if recomputed["triggered_rules"] != saved.get("triggered_rules"):
        raise ValueError(f"Saved triggered rules do not reconstruct for {condition[0]} severity {condition[1]}.")

def _rule_signal(value: float, rule: str, trust_policy: dict) -> str:
    if value >= trust_policy["do_not_trust"][rule]:
        return "do_not_trust"
    if value >= trust_policy["caution"][rule]:
        return "caution"
    return "trust"

def build_trust_rule_attribution(metrics: pd.DataFrame, trust_policy: dict, saved_trust_records: list[dict]) -> pd.DataFrame:
    """Record how every active rule contributes to each warning."""
    required = {"dataset", "model", "seed", "degradation", "severity", "accuracy", "ece", "confidence_accuracy_gap", "hcer_fixed", "hcer_adaptive"}
    _require_columns(metrics, required, "Metrics summary")
    model = _single_model(metrics, "Metrics summary")
    baseline_rows = metrics[(metrics["degradation"] == "none") & (metrics["severity"] == 0)]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one undegraded baseline row.")

    baseline = baseline_rows.iloc[0].to_dict()
    active_metrics = list(trust_policy["active_metrics"])
    saved = pd.DataFrame(saved_trust_records)
    saved_required = {"dataset", "model", "degradation", "severity", "trust_signal", "triggered_rules"}
    _require_columns(saved, saved_required, "Saved trust signal")
    if _single_model(saved, "Saved trust signal") != model:
        raise ValueError("Saved trust signal and metrics summary describe different models.")
    if saved.duplicated(["degradation", "severity"]).any():
        raise ValueError("Saved trust signal must contain one row per condition.")

    saved_lookup = saved.set_index(["degradation", "severity"]).to_dict("index")
    rows = []

    for _, condition_metrics in metrics.iterrows():
        condition = (condition_metrics["degradation"], int(condition_metrics["severity"]))

        if condition not in saved_lookup:
            raise ValueError(f"Saved trust signal is missing {condition[0]} severity {condition[1]}.")
        
        # recalculate the warning so each rule can be tied back to the result that was actually saved
        record = assign_trust_signal(condition_metrics.to_dict(), baseline, trust_policy)
        _validate_saved_trust(record, saved_lookup[condition], condition)

        for rule in active_metrics:
            value = float(record[rule])
            rule_signal = _rule_signal(value, rule, trust_policy)

            if rule in PERFORMANCE_METRICS:
                channel = "performance"
                channel_signal = record["performance_signal"]
            elif rule in CONFIDENCE_METRICS:
                channel = "confidence"
                channel_signal = record["confidence_signal"]
            else:
                channel = "diagnostic"
                channel_signal = "not_applicable"

            rows.append({
                "dataset": record["dataset"],
                "model": model,
                "seed": int(condition_metrics["seed"]),
                "degradation": record["degradation"],
                "severity": record["severity"],
                "trust_signal": record["trust_signal"],
                "performance_signal": record["performance_signal"],
                "confidence_signal": record["confidence_signal"],
                "rule_channel": channel,
                "rule": rule,
                "rule_label": RULE_LABELS[rule],
                "deterioration_value": value,
                "caution_threshold": float(trust_policy["caution"][rule]),
                "do_not_trust_threshold": float(trust_policy["do_not_trust"][rule]),
                "rule_signal": rule_signal,
                "triggered": rule_signal != "trust",
                "is_channel_driver": rule_signal != "trust" and rule_signal == channel_signal,
                "is_overall_driver": rule_signal != "trust" and rule_signal == record["trust_signal"],
                "gap_direction": record["gap_direction"] if rule == "gap_deterioration" else None
            })

    return _sort_conditions(pd.DataFrame(rows), ["rule_channel", "rule"])

def _policy_for_rules(trust_policy: dict, active_rules: list[str]) -> dict:
    return {
        **trust_policy,
        "active_metrics": active_rules,
        "caution": {rule: trust_policy["caution"][rule] for rule in active_rules},
        "do_not_trust": {rule: trust_policy["do_not_trust"][rule] for rule in active_rules}
    }

def _ablation_scenarios(active_rules: list[str]) -> list[dict]:
    scenarios = []
    # remove one rule at a time to see whether any warning actually depends on it
    for removed_rule in active_rules:
        retained = [rule for rule in active_rules if rule != removed_rule]
        scenarios.append({
            "analysis_type": "leave_one_rule_out",
            "configuration": f"without_{removed_rule}",
            "rules_retained": retained,
            "rules_removed": [removed_rule]
        })
    # then isolate each rule to see how much of the full warning it can reproduce alone
    for retained_rule in active_rules:
        scenarios.append({
            "analysis_type": "single_rule",
            "configuration": f"only_{retained_rule}",
            "rules_retained": [retained_rule],
            "rules_removed": [rule for rule in active_rules if rule != retained_rule]
        })
    # finally, compare the performance and confidence rule groups as two separate views
    performance_rules = [rule for rule in active_rules if rule in PERFORMANCE_METRICS]
    confidence_rules = [rule for rule in active_rules if rule in CONFIDENCE_METRICS]

    scenarios.extend([
        {
            "analysis_type": "rule_channel",
            "configuration": "performance_rules",
            "rules_retained": performance_rules,
            "rules_removed": [rule for rule in active_rules if rule not in performance_rules]
        },
        {
            "analysis_type": "rule_channel",
            "configuration": "confidence_rules",
            "rules_retained": confidence_rules,
            "rules_removed": [rule for rule in active_rules if rule not in confidence_rules]
        }
    ])

    return [scenario for scenario in scenarios if scenario["rules_retained"]]

def build_trust_rule_ablation(metrics: pd.DataFrame, trust_policy: dict) -> pd.DataFrame:
    """Compare the full warning against smaller active-rule sets."""
    required = {"dataset", "model", "seed", "degradation", "severity", "accuracy", "ece", "confidence_accuracy_gap", "hcer_fixed", "hcer_adaptive"}
    _require_columns(metrics, required, "Metrics summary")
    model = _single_model(metrics, "Metrics summary")
    baseline_rows = metrics[(metrics["degradation"] == "none") & (metrics["severity"] == 0)]

    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one undegraded baseline row.")

    baseline = baseline_rows.iloc[0].to_dict()
    active_rules = list(trust_policy["active_metrics"])
    scenarios = _ablation_scenarios(active_rules)
    rows = []

    for _, condition_metrics in metrics.iterrows():
        condition = condition_metrics.to_dict()
        full = assign_trust_signal(condition, baseline, trust_policy)

        for scenario in scenarios:
            ablated_policy = _policy_for_rules(trust_policy, scenario["rules_retained"])
            ablated = assign_trust_signal(condition, baseline, ablated_policy)

            rows.append({
                "dataset": full["dataset"],
                "model": model,
                "seed": int(condition_metrics["seed"]),
                "degradation": full["degradation"],
                "severity": full["severity"],
                "full_trust_signal": full["trust_signal"],
                "full_performance_signal": full["performance_signal"],
                "full_confidence_signal": full["confidence_signal"],
                "analysis_type": scenario["analysis_type"],
                "configuration": scenario["configuration"],
                "rules_retained": "|".join(scenario["rules_retained"]),
                "rules_removed": "|".join(scenario["rules_removed"]),
                "ablated_trust_signal": ablated["trust_signal"],
                "ablated_performance_signal": ablated["performance_signal"],
                "ablated_confidence_signal": ablated["confidence_signal"],
                "changed_from_full": ablated["trust_signal"] != full["trust_signal"],
                "performance_changed_from_full": ablated["performance_signal"] != full["performance_signal"],
                "confidence_changed_from_full": ablated["confidence_signal"] != full["confidence_signal"]
            })

    return _sort_conditions(pd.DataFrame(rows), ["analysis_type", "configuration"])

def load_trust_records(path) -> list[dict]:
    """Load saved trust records for offline analysis."""
    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list) or not records:
        raise ValueError("Trust signal must contain a non-empty list.")

    return records