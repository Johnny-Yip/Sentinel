"""Write the normalized Sentinel unified evaluation artifact set."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.evaluation.explain import build_explanation_summary
from sentinel.evaluation.experiment import EvaluationReport


REPORT_FILENAMES = {
    "metrics": "metrics.json",
    "model_comparison": "model_comparison.csv",
    "feature_importance": "feature_importance.csv",
    "confusion_matrix": "confusion_matrix.png",
    "summary": "report.md",
    "risk_ranking": "risk_ranking.csv",
    "risk_summary": "risk_summary.json",
    "prediction_explanations": "prediction_explanations.csv",
    "explanation_summary": "explanation_summary.json",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metric(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    escaped = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in escaped),
    ]


def _comparison_table(frame: pd.DataFrame) -> list[str]:
    if "accuracy" in frame:
        headers = [
            "Model",
            "Selected",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC",
        ]
        rows = [
            [
                str(row.model),
                "yes" if row.selected_model else "no",
                _metric(row.accuracy),
                _metric(row.precision),
                _metric(row.recall),
                _metric(row.f1),
                _metric(row.roc_auc),
                _metric(row.pr_auc),
            ]
            for row in frame.itertuples(index=False)
        ]
    else:
        headers = [
            "Model",
            "Folds",
            "Selected folds",
            "Mean accuracy",
            "Mean precision",
            "Mean recall",
            "Mean F1",
            "Mean ROC-AUC",
            "Mean PR-AUC",
        ]
        rows = [
            [
                str(row.model),
                str(row.fold_count),
                str(row.selected_fold_count),
                _metric(row.mean_accuracy),
                _metric(row.mean_precision),
                _metric(row.mean_recall),
                _metric(row.mean_f1),
                _metric(row.mean_roc_auc),
                _metric(row.mean_pr_auc),
            ]
            for row in frame.itertuples(index=False)
        ]
    return _markdown_table(headers, rows)


def _importance_table(frame: pd.DataFrame) -> list[str]:
    top = frame.loc[frame["rank"] <= 3].sort_values(
        ["model", "rank"], kind="stable"
    )
    return _markdown_table(
        ["Model", "Rank", "Feature", "Importance", "Direction", "Method"],
        [
            [
                str(row.model),
                str(row.rank),
                str(row.feature),
                _metric(row.importance),
                str(row.direction),
                str(row.method),
            ]
            for row in top.itertuples(index=False)
        ],
    )


def build_markdown_report(report: EvaluationReport) -> str:
    """Build a concise report from the same normalized data saved to CSV/JSON."""
    dataset = report.dataset_summary
    explanation_summary = report.explanation_summary or build_explanation_summary(
        report.prediction_explanations
    )
    lines = [
        "# Sentinel V6 evaluation report",
        "",
        "## Experiment",
        "",
        f"- Type: `{report.experiment_type}`",
        f"- Selected model: `{report.selected_model}`",
        f"- Selection metric: `{report.selection_metric}`",
        f"- Snapshots: {dataset['snapshots']:,}",
        f"- Repositories: {dataset['repository_count']}",
        f"- Positive labels: {dataset['positive_labels']:,} "
        f"({_metric(dataset['positive_rate'])})",
        f"- Date range: {dataset['date_range']['start']} to "
        f"{dataset['date_range']['end']}",
        "",
        "## Model comparison",
        "",
        *_comparison_table(report.model_comparison),
        "",
        "## Top feature importance",
        "",
        *_importance_table(report.feature_importance),
        "",
        "## Key findings",
        "",
        *(f"- {finding}" for finding in report.key_findings),
        "",
        "## Risk ranking",
        "",
        f"- Evaluated samples: {report.risk_summary['total_samples']:,}",
        f"- High-risk samples at score >= "
        f"{_metric(report.risk_summary['risk_threshold'])}: "
        f"{report.risk_summary['high_risk_sample_count']:,}",
        f"- Mean risk score: {_metric(report.risk_summary['mean_risk_score'])}",
        f"- Maximum risk score: {_metric(report.risk_summary['max_risk_score'])}",
        "",
        "## Individual prediction explanations",
        "",
        f"- Explained samples: "
        f"{explanation_summary['total_explained_samples']:,}",
        f"- Feature contributions: "
        f"{explanation_summary['total_explained_feature_contributions']:,}",
        "- Positive contributions increase predicted defect risk; negative "
        "contributions decrease it.",
        "",
        "## Artifacts",
        "",
        "Machine-readable metrics and comparisons are in `metrics.json`, "
        "`model_comparison.csv`, and `feature_importance.csv`. The selected "
        "evaluation confusion matrix is in `confusion_matrix.png`. Sample-level "
        "risk ordering and its summary are in `risk_ranking.csv` and "
        "`risk_summary.json`. Local feature-level contributions and their "
        "aggregate summary are in `prediction_explanations.csv` and "
        "`explanation_summary.json`.",
        "",
    ]
    return "\n".join(lines)


def _save_confusion_matrix(
    matrix: np.ndarray, label: str, destination: Path
) -> None:
    # Import lazily so metric-only library use and CLI help do not initialize
    # Matplotlib or its font cache.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    values = np.asarray(matrix, dtype=int)
    if values.shape != (2, 2):
        raise ValueError("A binary confusion matrix must have shape (2, 2).")

    figure = Figure(figsize=(5.6, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion matrix\n{label}",
    )
    midpoint = (float(values.max()) + float(values.min())) / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{values[row, column]:,}",
                ha="center",
                va="center",
                color="white" if values[row, column] > midpoint else "black",
                fontsize=13,
            )
    figure.savefig(destination, dpi=150, format="png")


def save_evaluation_report(
    report: EvaluationReport, output_dir: str | Path
) -> dict[str, Path]:
    """Persist the unified report contract in one experiment directory."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        name: destination / filename
        for name, filename in REPORT_FILENAMES.items()
    }
    metrics = {
        "experiment_type": report.experiment_type,
        "selected_model": report.selected_model,
        "selection_metric": report.selection_metric,
        "dataset": report.dataset_summary,
        "confusion_matrix": report.confusion_matrix,
        "confusion_matrix_scope": report.confusion_matrix_label,
        **report.metrics,
    }
    paths["metrics"].write_text(
        json.dumps(_json_safe(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.model_comparison.to_csv(paths["model_comparison"], index=False)
    report.feature_importance.to_csv(paths["feature_importance"], index=False)
    report.risk_ranking.to_csv(paths["risk_ranking"], index=False)
    paths["risk_summary"].write_text(
        json.dumps(_json_safe(report.risk_summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.prediction_explanations.to_csv(
        paths["prediction_explanations"], index=False
    )
    explanation_summary = report.explanation_summary or build_explanation_summary(
        report.prediction_explanations
    )
    paths["explanation_summary"].write_text(
        json.dumps(_json_safe(explanation_summary), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    _save_confusion_matrix(
        report.confusion_matrix,
        report.confusion_matrix_label,
        paths["confusion_matrix"],
    )
    paths["summary"].write_text(
        build_markdown_report(report), encoding="utf-8"
    )
    return paths
