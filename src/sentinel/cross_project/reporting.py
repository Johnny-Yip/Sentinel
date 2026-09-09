"""CLI and artifact reporting for Sentinel V4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.cross_project.experiment import CrossProjectResult
from sentinel.ml.models import MODEL_ORDER
from sentinel.ml.reporting import MODEL_LABELS


def _metric(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def print_repository_summaries(result: CrossProjectResult) -> None:
    print("\nRepository datasets")
    for summary in result.dataset.summaries:
        print(
            f"  {summary.repository}: snapshots={summary.snapshot_count:,}, "
            f"files={summary.unique_java_files:,}, positive={summary.positive_labels:,}, "
            f"negative={summary.negative_labels:,}, "
            f"positive_rate={summary.positive_rate:.4f}, "
            f"dates={summary.start_date} to {summary.end_date}"
        )
        print(f"    CSV: {summary.feature_csv_path}")


def print_cross_project_report(result: CrossProjectResult) -> None:
    """Print fold-level choices, held-out metrics, aggregate results, and audit."""
    print("\nLeave-one-project-out folds")
    for evaluation, selection in zip(
        result.fold_evaluations, result.model_selection, strict=True
    ):
        fold = evaluation.fold
        print(f"\n  Held out: {fold.held_out_repository}")
        print(f"  Train/validation repositories: {', '.join(fold.training_repositories)}")
        print(
            f"  Rows: train={len(fold.train):,}, validation={len(fold.validation):,}, "
            f"held-out={len(fold.test):,}"
        )
        print("  Validation selection and held-out performance:")
        for model_name in MODEL_ORDER:
            candidate = selection["candidates"][model_name]
            marker = " [selected]" if model_name == selection["selected_model"] else ""
            print(
                f"    {MODEL_LABELS[model_name]}{marker}: "
                f"validation PR-AUC={_metric(candidate['validation_pr_auc'])}, "
                f"selected threshold={candidate['selected_threshold']:.6f}"
            )
            rows = result.folds.loc[
                (result.folds["held_out_repository"] == fold.held_out_repository)
                & (result.folds["model"] == model_name)
            ]
            for row in rows.itertuples(index=False):
                strategy = (
                    "0.5" if row.threshold_strategy == "default_0_5" else "selected"
                )
                print(
                    f"      test @ {strategy}: precision={_metric(row.precision)}, "
                    f"recall={_metric(row.recall)}, F1={_metric(row.f1)}, "
                    f"ROC-AUC={_metric(row.roc_auc)}, PR-AUC={_metric(row.pr_auc)}, "
                    f"lift={_metric(row.pr_auc_lift)}, pred+={_metric(row.positive_prediction_rate)}, "
                    f"CM=[[{row.confusion_tn}, {row.confusion_fp}], "
                    f"[{row.confusion_fn}, {row.confusion_tp}]]"
                )

    print("\nAggregate held-out metrics (validation-selected thresholds)")
    for row in result.aggregate.itertuples(index=False):
        print(
            f"  {MODEL_LABELS[row.model]}: mean PR-AUC={_metric(row.mean_pr_auc)}, "
            f"median={_metric(row.median_pr_auc)}, std={_metric(row.std_pr_auc)}, "
            f"range={_metric(row.min_pr_auc)}..{_metric(row.max_pr_auc)}, "
            f"mean lift={_metric(row.mean_pr_auc_lift)}, mean F1={_metric(row.mean_f1)}, "
            f"mean precision={_metric(row.mean_precision)}, "
            f"mean recall={_metric(row.mean_recall)}, "
            f"mean ROC-AUC={_metric(row.mean_roc_auc)}"
        )

    interpretation = result.report["generalization_interpretation"]
    print("\nGeneralization summary")
    print(
        "  Selected models above prevalence on every repository: "
        f"{interpretation['selected_models_above_prevalence_on_all_repositories']}"
    )
    print(
        "  Repositories above prevalence: "
        f"{', '.join(interpretation['repositories_above_prevalence']) or 'none'}"
    )
    print(
        "  Repositories not above prevalence: "
        f"{', '.join(interpretation['repositories_not_above_prevalence']) or 'none'}"
    )
    print(
        f"  Best mean PR-AUC model: {interpretation['best_mean_pr_auc_model']}; "
        f"most stable non-dummy: {interpretation['most_stable_non_dummy_model']}; "
        "Logistic Regression most robust: "
        f"{interpretation['logistic_regression_remains_most_robust']}"
    )
    comparison = result.report["same_project_comparison"]
    if comparison.get("available"):
        logistic = comparison["comparisons"].get("logistic_regression")
        if logistic:
            delta = logistic["cross_minus_same"]
            print(
                "  commons-lang Logistic Regression cross minus same project: "
                f"PR-AUC={_metric(delta['pr_auc'])}, F1={_metric(delta['f1'])}, "
                f"ROC-AUC={_metric(delta['roc_auc'])}"
            )
    else:
        print(f"  Same-project comparison unavailable: {comparison['reason']}")

    coefficients = result.coefficient_stability
    positive = coefficients.loc[
        coefficients["direction"] == "consistently_positive", "feature"
    ].tolist()
    negative = coefficients.loc[
        coefficients["direction"] == "consistently_negative", "feature"
    ].tolist()
    print(
        "  Consistent Logistic Regression positive signals: "
        f"{', '.join(positive) or 'none'}"
    )
    print(
        "  Consistent Logistic Regression negative signals: "
        f"{', '.join(negative) or 'none'}"
    )

    print("\nDataset shift (largest absolute standardized median shift per repository)")
    for repository, rows in result.dataset_shift.groupby("held_out_repository"):
        ranked = rows.dropna(subset=["absolute_standardized_median_shift"]).sort_values(
            "absolute_standardized_median_shift", ascending=False
        )
        if ranked.empty:
            value = "no finite score (training IQRs were zero)"
        else:
            top = ranked.iloc[0]
            value = (
                f"{top['feature']}={top['standardized_median_shift']:.4f} "
                "training-IQRs"
            )
        print(f"  {repository}: {value}")

    print(
        "\nCross-project leakage audit: "
        f"{result.report['leakage_audit']['status'].upper()}"
    )


def build_markdown_report(result: CrossProjectResult) -> str:
    """Build a compact human-readable report backed by the CSV/JSON artifacts."""
    lines = [
        "# Sentinel V4 cross-project report",
        "",
        "This report uses leave-one-project-out evaluation. Each held-out repository "
        "is absent from training, preprocessing, threshold tuning, and model selection.",
        "",
        "## Repository datasets",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                "Repository",
                "Snapshots",
                "Files",
                "Positive",
                "Negative",
                "Positive rate",
                "Date range",
                "CSV",
            ],
            [
                [
                    summary.repository,
                    str(summary.snapshot_count),
                    str(summary.unique_java_files),
                    str(summary.positive_labels),
                    str(summary.negative_labels),
                    f"{summary.positive_rate:.4f}",
                    f"{summary.start_date} to {summary.end_date}",
                    summary.feature_csv_path,
                ]
                for summary in result.dataset.summaries
            ],
        )
    )
    lines.extend(["", "## Per-repository held-out results", ""])
    selected = result.folds.loc[
        result.folds["threshold_strategy"] == "validation_selected"
    ]
    lines.extend(
        _markdown_table(
            [
                "Held out",
                "Model",
                "Selected",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
                "PR-AUC",
                "Prevalence",
                "Lift",
            ],
            [
                [
                    row.held_out_repository,
                    MODEL_LABELS[row.model],
                    "yes" if row.is_selected_model else "no",
                    f"{row.threshold:.4f}",
                    _metric(row.precision),
                    _metric(row.recall),
                    _metric(row.f1),
                    _metric(row.roc_auc),
                    _metric(row.pr_auc),
                    _metric(row.class_prevalence),
                    _metric(row.pr_auc_lift),
                ]
                for row in selected.itertuples(index=False)
            ],
        )
    )
    lines.extend(["", "## Aggregate results", ""])
    lines.extend(
        _markdown_table(
            [
                "Model",
                "Mean PR-AUC",
                "Median",
                "Std",
                "Min",
                "Max",
                "Mean lift",
                "Mean F1",
                "Mean precision",
                "Mean recall",
                "Mean ROC-AUC",
            ],
            [
                [
                    MODEL_LABELS[row.model],
                    _metric(row.mean_pr_auc),
                    _metric(row.median_pr_auc),
                    _metric(row.std_pr_auc),
                    _metric(row.min_pr_auc),
                    _metric(row.max_pr_auc),
                    _metric(row.mean_pr_auc_lift),
                    _metric(row.mean_f1),
                    _metric(row.mean_precision),
                    _metric(row.mean_recall),
                    _metric(row.mean_roc_auc),
                ]
                for row in result.aggregate.itertuples(index=False)
            ],
        )
    )

    interpretation = result.report["generalization_interpretation"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Above prevalence on every held-out repository: "
            f"{interpretation['selected_models_above_prevalence_on_all_repositories']}.",
            f"- Repositories above prevalence: "
            f"{', '.join(interpretation['repositories_above_prevalence']) or 'none'}.",
            f"- Repositories not above prevalence: "
            f"{', '.join(interpretation['repositories_not_above_prevalence']) or 'none'}.",
            f"- Best mean PR-AUC model: {interpretation['best_mean_pr_auc_model']}.",
            f"- Most stable non-dummy model: "
            f"{interpretation['most_stable_non_dummy_model']} "
            f"(PR-AUC std={interpretation['most_stable_non_dummy_pr_auc_std']:.4f}).",
            f"- Best-model PR-AUC range across repositories: "
            f"{interpretation['best_model_pr_auc_range_across_repositories']:.4f}.",
            "",
        ]
    )

    comparison = result.report["same_project_comparison"]
    lines.extend(["## Same-project versus cross-project commons-lang", ""])
    if comparison.get("available"):
        logistic = comparison["comparisons"].get("logistic_regression")
        if logistic:
            lines.append(
                "For Logistic Regression, cross-project minus same-project was "
                f"PR-AUC {_metric(logistic['cross_minus_same']['pr_auc'])}, "
                f"F1 {_metric(logistic['cross_minus_same']['f1'])}, and ROC-AUC "
                f"{_metric(logistic['cross_minus_same']['roc_auc'])}. Cross-project "
                "evaluation is harder because the model has never seen the target "
                "repository during fitting or selection."
            )
    else:
        lines.append(f"Comparison unavailable: {comparison['reason']}.")

    lines.extend(
        [
            "",
            "## Leakage audit",
            "",
            f"Overall status: **{result.report['leakage_audit']['status'].upper()}**.",
            "",
            "Full coefficients, feature importances, dataset-shift statistics, "
            "default-threshold results, fold definitions, and audit checks are in "
            "the companion CSV and JSON artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def save_cross_project_artifacts(
    result: CrossProjectResult, output_dir: str | Path
) -> dict[str, Path]:
    """Persist every requested V4 result without modifying V3 artifacts."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "folds": destination / "folds.csv",
        "aggregate": destination / "aggregate.csv",
        "model_selection": destination / "model_selection.json",
        "selected_thresholds": destination / "selected_thresholds.csv",
        "coefficient_stability": destination / "coefficient_stability.csv",
        "feature_importance_stability": destination
        / "feature_importance_stability.csv",
        "dataset_shift": destination / "dataset_shift.csv",
        "report_json": destination / "report.json",
        "report_markdown": destination / "report.md",
    }
    result.folds.to_csv(paths["folds"], index=False)
    result.aggregate.to_csv(paths["aggregate"], index=False)
    paths["model_selection"].write_text(
        json.dumps(result.model_selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result.folds.loc[
        result.folds["threshold_strategy"] == "validation_selected",
        [
            "held_out_repository",
            "model",
            "is_selected_model",
            "selected_validation_threshold",
            "validation_pr_auc",
            "validation_f1",
        ],
    ].to_csv(paths["selected_thresholds"], index=False)
    result.coefficient_stability.to_csv(paths["coefficient_stability"], index=False)
    result.feature_importance_stability.to_csv(
        paths["feature_importance_stability"], index=False
    )
    result.dataset_shift.to_csv(paths["dataset_shift"], index=False)
    paths["report_json"].write_text(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["report_markdown"].write_text(
        build_markdown_report(result), encoding="utf-8"
    )
    return paths
