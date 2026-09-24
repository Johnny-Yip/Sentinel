"""Write the normalized Sentinel unified evaluation artifact set."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel.evaluation.action_evaluation import (
    ACTION_QUALITY_COLUMNS,
    ActionEvaluationResult,
    action_evaluation_markdown_lines,
    build_action_evaluation,
    build_action_evaluation_markdown,
)
from sentinel.evaluation.calibration import (
    CALIBRATION_SECTION,
    RiskCalibrationResult,
    build_risk_calibration,
    calibration_markdown_lines,
)
from sentinel.evaluation.decision_brief import (
    DEVELOPER_ACTION_COLUMNS,
    DecisionBriefResult,
    build_decision_brief,
    build_decision_brief_markdown,
    decision_brief_markdown_lines,
)
from sentinel.evaluation.explain import build_explanation_summary
from sentinel.evaluation.experiment import EvaluationReport
from sentinel.evaluation.insights import build_actionable_insights
from sentinel.evaluation.project_intelligence import (
    DEVELOPER_PRIORITY_COLUMNS,
    ProjectIntelligenceResult,
    build_project_intelligence,
)


CALIBRATION_FILENAMES = {
    "calibration": "calibration.csv",
    "threshold_analysis": "threshold_analysis.csv",
    "calibration_summary": "calibration_summary.json",
}

REPORT_FILENAMES = {
    **CALIBRATION_FILENAMES,
    "metrics": "metrics.json",
    "model_comparison": "model_comparison.csv",
    "feature_importance": "feature_importance.csv",
    "confusion_matrix": "confusion_matrix.png",
    "summary": "report.md",
    "risk_ranking": "risk_ranking.csv",
    "risk_summary": "risk_summary.json",
    "prediction_explanations": "prediction_explanations.csv",
    "explanation_summary": "explanation_summary.json",
    "actionable_insights": "actionable_insights.json",
    "project_intelligence": "project_intelligence.json",
    "developer_priority": "developer_priority.csv",
    "developer_risk_brief_json": "developer_risk_brief.json",
    "developer_risk_brief_markdown": "developer_risk_brief.md",
    "developer_actions": "developer_actions.csv",
    "explanation_action_evaluation_json": "explanation_action_evaluation.json",
    "explanation_action_evaluation_markdown": "explanation_action_evaluation.md",
    "action_quality": "action_quality.csv",
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


def _percentage_metric(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.2f}%"


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


def _resolved_project_intelligence(
    report: EvaluationReport,
) -> ProjectIntelligenceResult:
    if report.project_intelligence:
        queue = report.developer_priority.copy()
        if queue.empty:
            queue = pd.DataFrame(columns=DEVELOPER_PRIORITY_COLUMNS)
        return ProjectIntelligenceResult(report.project_intelligence, queue)
    return build_project_intelligence(
        report.risk_ranking,
        report.prediction_explanations,
        report.actionable_insights,
        risk_threshold=float(report.risk_summary["risk_threshold"]),
        experiment_type=report.experiment_type,
    )


def _resolved_decision_brief(
    report: EvaluationReport,
    project_intelligence: ProjectIntelligenceResult,
) -> DecisionBriefResult:
    if report.decision_brief:
        actions = report.developer_actions.copy()
        if actions.empty:
            actions = pd.DataFrame(columns=DEVELOPER_ACTION_COLUMNS)
        return DecisionBriefResult(report.decision_brief, actions)
    return build_decision_brief(
        report.risk_ranking,
        report.prediction_explanations,
        report.actionable_insights,
        project_intelligence.artifact,
        project_intelligence.developer_priority,
        risk_threshold=float(report.risk_summary["risk_threshold"]),
        experiment_type=report.experiment_type,
        model_metadata=report.risk_summary.get("model_metadata", {}),
    )


def _resolved_action_evaluation(
    report: EvaluationReport,
    decision_brief: DecisionBriefResult,
) -> ActionEvaluationResult:
    """Evaluate retained artifacts for reports created before Phase 6."""
    artifact = getattr(report, "explanation_action_evaluation", {})
    if artifact:
        quality = getattr(report, "action_quality", pd.DataFrame()).copy()
        if quality.empty:
            quality = pd.DataFrame(columns=ACTION_QUALITY_COLUMNS)
        return ActionEvaluationResult(artifact, quality)
    return build_action_evaluation(
        report.risk_ranking,
        report.prediction_explanations,
        decision_brief.developer_actions,
        risk_threshold=float(report.risk_summary["risk_threshold"]),
        experiment_type=report.experiment_type,
    )


def _resolved_calibration(report: EvaluationReport) -> RiskCalibrationResult:
    """Support reports constructed before Phase 7 without model access."""
    result = getattr(report, "risk_calibration", None)
    return result if result is not None else build_risk_calibration(
        report.risk_ranking, experiment_type=report.experiment_type
    )


def save_risk_calibration(
    result: RiskCalibrationResult, output_dir: str | Path,
) -> dict[str, Path]:
    """Save Phase 7 only, appending/replacing its section in report.md.

    This supports artifact-only analysis of existing Phase 1–6 output directories.
    Earlier report sections and artifacts are retained.
    """
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {name: destination / filename for name, filename in CALIBRATION_FILENAMES.items()}
    paths["calibration_summary"].write_text(
        json.dumps(result.artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result.calibration.to_csv(paths["calibration"], index=False)
    result.threshold_analysis.to_csv(paths["threshold_analysis"], index=False)
    paths["summary"] = destination / "report.md"
    previous = paths["summary"].read_text(encoding="utf-8") if paths["summary"].exists() else ""
    lines = previous.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.rstrip() == CALIBRATION_SECTION), None)
    section = "\n".join(calibration_markdown_lines(result.artifact))
    if start is None:
        markdown = previous + ("\n\n" if previous else "") + section
    else:
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        suffix = "".join(lines[end:])
        markdown = "".join(lines[:start]) + section + ("\n" + suffix if suffix else "")
    paths["summary"].write_text(markdown, encoding="utf-8")
    return paths


def _signal_names(records: list[dict[str, Any]]) -> str:
    return ", ".join(str(record["feature"]) for record in records[:3]) or "none"


def _project_intelligence_lines(
    intelligence: dict[str, Any], priority: pd.DataFrame
) -> list[str]:
    if intelligence["summary"].get("analysis_scope") == "per_project":
        analyses = intelligence.get("projects", [])
        lines = [
            "Cross-project intelligence is kept separate for each held-out project.",
            "",
        ]
        if analyses:
            lines.extend(
                _markdown_table(
                    [
                        "Project",
                        "Samples",
                        "Mean risk",
                        "High-risk rate",
                        "Top 10% risk share",
                        "Dominant signal",
                        "Temporal direction",
                    ],
                    [
                        [
                            str(analysis["project"]),
                            str(analysis["summary"]["total_evaluated_samples"]),
                            _metric(analysis["summary"]["mean_predicted_risk"]),
                            _percentage_metric(
                                analysis["summary"]["high_risk_percentage"]
                            ),
                            _percentage_metric(
                                analysis["risk_concentration"][
                                    "top_10_percent_risk_contribution_percentage"
                                ]
                            ),
                            _signal_names(
                                analysis["summary"]["dominant_risk_signals"]
                            ),
                            str(analysis["temporal_analysis"]["direction"]),
                        ]
                        for analysis in analyses
                    ],
                )
            )
    else:
        summary = intelligence["summary"]
        concentration = intelligence["risk_concentration"]
        temporal = intelligence["temporal_analysis"]
        lines = [
            f"- Overall fitted-model risk: mean "
            f"{_metric(summary['mean_predicted_risk'])}; "
            f"{summary['high_risk_sample_count']:,} of "
            f"{summary['total_evaluated_samples']:,} samples "
            f"({_percentage_metric(summary['high_risk_percentage'])}) are at or "
            f"above the configured risk cutoff.",
            f"- Concentration: the top 10% of samples contribute "
            f"{_percentage_metric(concentration['top_10_percent_risk_contribution_percentage'])} "
            "of aggregate predicted risk; "
            f"{concentration['samples_responsible_for_50_percent_of_predicted_risk']:,} "
            "samples contribute its first 50%.",
            f"- Dominant risk signals: "
            f"{_signal_names(summary['dominant_risk_signals'])}.",
            f"- Temporal direction: {temporal['direction']}"
            + ("." if temporal["supported"] else " (timestamps unavailable)."),
        ]
        hotspots = intelligence.get("hotspots", [])[:3]
        if hotspots:
            lines.append(
                "- Top recurring hotspots: "
                + ", ".join(
                    f"`{row['identifier']}` ({row['high_risk_count']} high-risk)"
                    for row in hotspots
                )
                + "."
            )

    if not priority.empty:
        lines.extend(
            [
                "",
                "Top inspection priorities:",
                "",
                *_markdown_table(
                    ["Project", "Rank", "Identifier", "Risk", "Priority score"],
                    [
                        [
                            str(row.project),
                            str(row.rank),
                            str(row.identifier) or "unavailable",
                            _metric(row.predicted_risk),
                            _metric(row.priority_score),
                        ]
                        for row in priority.groupby("project", sort=False)
                        .head(3)
                        .itertuples(index=False)
                    ],
                ),
            ]
        )
    return lines


def build_markdown_report(
    report: EvaluationReport,
    *,
    project_intelligence: dict[str, Any] | None = None,
    developer_priority: pd.DataFrame | None = None,
    decision_brief: dict[str, Any] | None = None,
    explanation_action_evaluation: dict[str, Any] | None = None,
    calibration_summary: dict[str, Any] | None = None,
) -> str:
    """Build a concise report from the same normalized data saved to CSV/JSON."""
    dataset = report.dataset_summary
    explanation_summary = report.explanation_summary or build_explanation_summary(
        report.prediction_explanations
    )
    actionable_insights = report.actionable_insights or build_actionable_insights(
        report.prediction_explanations
    )
    if project_intelligence is None or developer_priority is None:
        resolved = _resolved_project_intelligence(report)
        project_intelligence = resolved.artifact
        developer_priority = resolved.developer_priority
    if decision_brief is None or explanation_action_evaluation is None:
        intelligence_result = ProjectIntelligenceResult(
            project_intelligence, developer_priority
        )
        decision_result = _resolved_decision_brief(
            report, intelligence_result
        )
        if decision_brief is None:
            decision_brief = decision_result.artifact
        if explanation_action_evaluation is None:
            explanation_action_evaluation = _resolved_action_evaluation(
                report, decision_result
            ).artifact
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
        "## Actionable risk insights",
        "",
        f"- Samples with risk signals: "
        f"{actionable_insights['samples_with_risk_signals']:,}",
        f"- Samples with protective signals: "
        f"{actionable_insights['samples_with_protective_signals']:,}",
        "- Recommendations translate the strongest local contributions into "
        "developer inspection prompts.",
        "",
        "## Project Risk Intelligence",
        "",
        *_project_intelligence_lines(project_intelligence, developer_priority),
        "",
        "## Artifacts",
        "",
        "Machine-readable metrics and comparisons are in `metrics.json`, "
        "`model_comparison.csv`, and `feature_importance.csv`. The selected "
        "evaluation confusion matrix is in `confusion_matrix.png`. Sample-level "
        "risk ordering and its summary are in `risk_ranking.csv` and "
        "`risk_summary.json`. Local feature-level contributions and their "
        "aggregate summary are in `prediction_explanations.csv` and "
        "`explanation_summary.json`. Developer-facing per-sample guidance and "
        "common signals are in `actionable_insights.json`. Project-level summaries "
        "are in `project_intelligence.json`, and the complete deterministic "
        "inspection queue is in `developer_priority.csv`. The final decision "
        "summary is in `developer_risk_brief.json` and "
        "`developer_risk_brief.md`; its high-risk inspection queue is in "
        "`developer_actions.csv`. Diagnostic explanation and action heuristics "
        "are in `explanation_action_evaluation.json` and "
        "`explanation_action_evaluation.md`; per-action scores and evidence flags "
        "are in `action_quality.csv`.",
        "",
        "## Developer Risk Brief",
        "",
        *decision_brief_markdown_lines(decision_brief, embedded=True),
        "",
        "## Explanation and Action Evaluation",
        "",
        *action_evaluation_markdown_lines(
            explanation_action_evaluation, embedded=True
        ),
        "",
    ]
    lines.extend(calibration_markdown_lines(
        calibration_summary if calibration_summary is not None
        else _resolved_calibration(report).artifact
    ))
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
    actionable_insights = report.actionable_insights or build_actionable_insights(
        report.prediction_explanations
    )
    paths["actionable_insights"].write_text(
        json.dumps(_json_safe(actionable_insights), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    project_intelligence = _resolved_project_intelligence(report)
    paths["project_intelligence"].write_text(
        json.dumps(
            _json_safe(project_intelligence.artifact), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    project_intelligence.developer_priority.to_csv(
        paths["developer_priority"], index=False
    )
    decision_brief = _resolved_decision_brief(report, project_intelligence)
    paths["developer_risk_brief_json"].write_text(
        json.dumps(_json_safe(decision_brief.artifact), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    paths["developer_risk_brief_markdown"].write_text(
        build_decision_brief_markdown(decision_brief.artifact),
        encoding="utf-8",
    )
    decision_brief.developer_actions.to_csv(
        paths["developer_actions"], index=False
    )
    action_evaluation = _resolved_action_evaluation(report, decision_brief)
    paths["explanation_action_evaluation_json"].write_text(
        json.dumps(
            _json_safe(action_evaluation.artifact),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["explanation_action_evaluation_markdown"].write_text(
        build_action_evaluation_markdown(action_evaluation.artifact),
        encoding="utf-8",
    )
    action_evaluation.action_quality.to_csv(paths["action_quality"], index=False)
    calibration = _resolved_calibration(report)
    paths["calibration_summary"].write_text(
        json.dumps(calibration.artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    calibration.calibration.to_csv(paths["calibration"], index=False)
    calibration.threshold_analysis.to_csv(paths["threshold_analysis"], index=False)
    _save_confusion_matrix(
        report.confusion_matrix,
        report.confusion_matrix_label,
        paths["confusion_matrix"],
    )
    paths["summary"].write_text(
        build_markdown_report(
            report,
            project_intelligence=project_intelligence.artifact,
            developer_priority=project_intelligence.developer_priority,
            decision_brief=decision_brief.artifact,
            explanation_action_evaluation=action_evaluation.artifact,
            calibration_summary=calibration.artifact,
        ),
        encoding="utf-8",
    )
    return paths
