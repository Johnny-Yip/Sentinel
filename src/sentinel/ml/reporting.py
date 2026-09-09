"""Concise CLI reporting for V3 training runs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from sentinel.ml.models import MODEL_ORDER
from sentinel.ml.splitting import TemporalSplits
from sentinel.ml.training import TrainingResult


MODEL_LABELS = {
    "dummy": "Dummy",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def _metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def print_split_report(splits: TemporalSplits) -> None:
    """Print temporal ranges and class distributions."""
    print("\nTemporal split (grouped by snapshot_date)")
    for name, summary in splits.summaries().items():
        print(
            f"  {name:<10} {summary.start_date} to {summary.end_date} | "
            f"rows={summary.rows:,} | positive={summary.positives:,} | "
            f"negative={summary.negatives:,} | "
            f"positive%={summary.positive_percentage:.2f}%"
        )


def _comparison_rows(result: TrainingResult) -> Iterable[list[str]]:
    for name in MODEL_ORDER:
        model = result.models[name]
        for split_name, metrics in (
            ("validation", model.validation_selected),
            ("test", model.test_selected),
        ):
            yield [
                MODEL_LABELS[name],
                split_name,
                f"{model.selected_threshold:.4f}",
                _metric(metrics["accuracy"]),
                _metric(metrics["precision"]),
                _metric(metrics["recall"]),
                _metric(metrics["f1"]),
                _metric(metrics["roc_auc"]),
                _metric(metrics["pr_auc"]),
                _metric(metrics["positive_prediction_rate"]),
            ]


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_confusion(label: str, metrics: dict[str, Any]) -> None:
    matrix = metrics["confusion_matrix"]
    print(
        f"    {label}: [[TN={matrix[0][0]}, FP={matrix[0][1]}], "
        f"[FN={matrix[1][0]}, TP={matrix[1][1]}]]"
    )


def print_model_report(result: TrainingResult) -> None:
    """Print thresholds, metrics, confusion matrices, and feature signals."""
    print("\nSelected-threshold comparison (PR-AUC is the primary metric)")
    _print_table(
        [
            "Model",
            "Split",
            "Thresh",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
            "Pred +",
        ],
        list(_comparison_rows(result)),
    )

    print("\nDefault 0.5 versus validation-selected thresholds")
    for name in MODEL_ORDER:
        model = result.models[name]
        print(f"  {MODEL_LABELS[name]} (selected threshold={model.selected_threshold:.6f})")
        for split_name, default, selected in (
            ("validation", model.validation_default, model.validation_selected),
            ("test", model.test_default, model.test_selected),
        ):
            print(
                f"    {split_name} @ 0.5: accuracy={_metric(default['accuracy'])}, "
                f"precision={_metric(default['precision'])}, "
                f"recall={_metric(default['recall'])}, f1={_metric(default['f1'])}, "
                f"ROC-AUC={_metric(default['roc_auc'])}, "
                f"PR-AUC={_metric(default['pr_auc'])}, "
                f"pred+={_metric(default['positive_prediction_rate'])}"
            )
            _print_confusion(f"{split_name} @ 0.5", default)
            print(
                f"    {split_name} @ selected: "
                f"accuracy={_metric(selected['accuracy'])}, "
                f"precision={_metric(selected['precision'])}, "
                f"recall={_metric(selected['recall'])}, f1={_metric(selected['f1'])}, "
                f"ROC-AUC={_metric(selected['roc_auc'])}, "
                f"PR-AUC={_metric(selected['pr_auc'])}, "
                f"pred+={_metric(selected['positive_prediction_rate'])}"
            )
            _print_confusion(f"{split_name} @ selected", selected)

        importance = model.feature_importance
        if "positive_coefficients" in importance:
            positive = ", ".join(
                f"{item['feature']}={item['value']:.4f}"
                for item in importance["positive_coefficients"]
            ) or "none"
            negative = ", ".join(
                f"{item['feature']}={item['value']:.4f}"
                for item in importance["negative_coefficients"]
            ) or "none"
            print(f"    strongest positive coefficients: {positive}")
            print(f"    strongest negative coefficients: {negative}")
        elif "feature_importances" in importance:
            values = ", ".join(
                f"{item['feature']}={item['value']:.4f}"
                for item in importance["feature_importances"]
            )
            print(f"    top feature importances: {values}")

    best = result.best_model
    print(
        f"\nBest model: {MODEL_LABELS[result.best_model_name]} selected by "
        f"validation PR-AUC={_metric(best.validation_selected['pr_auc'])}; "
        f"validation F1={_metric(best.validation_selected['f1'])}; "
        f"validation recall={_metric(best.validation_selected['recall'])}."
    )
