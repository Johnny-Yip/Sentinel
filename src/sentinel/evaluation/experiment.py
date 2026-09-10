"""Adapt existing within- and cross-project experiments to one report contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel import __version__
from sentinel.cross_project.data import MultiRepositoryDataset
from sentinel.cross_project.experiment import run_cross_project_evaluation
from sentinel.evaluation.explain import (
    EXPLANATION_COLUMNS,
    add_top_explanations_to_ranking,
    build_explanation_summary,
    explain_predictions,
    limit_explanations,
    sort_explanations,
)
from sentinel.evaluation.insights import (
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.evaluation.risk import (
    DEFAULT_RISK_THRESHOLD,
    DEFAULT_TOP_RISK,
    build_risk_ranking,
    build_risk_summary,
)
from sentinel.ml.data import (
    DATE_COLUMN,
    PATH_COLUMN,
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
)
from sentinel.ml.models import DEFAULT_RANDOM_STATE
from sentinel.ml.splitting import chronological_split
from sentinel.ml.training import ProgressCallback, TrainingResult, train_baselines


@dataclass
class EvaluationReport:
    """Normalized artifacts for one Sentinel unified evaluation experiment."""

    experiment_type: str
    dataset_summary: dict[str, Any]
    selected_model: str
    selection_metric: str
    metrics: dict[str, Any]
    model_comparison: pd.DataFrame
    feature_importance: pd.DataFrame
    confusion_matrix: np.ndarray
    confusion_matrix_label: str
    key_findings: tuple[str, ...]
    risk_ranking: pd.DataFrame
    risk_summary: dict[str, Any]
    prediction_explanations: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=EXPLANATION_COLUMNS)
    )
    explanation_summary: dict[str, Any] = field(default_factory=dict)
    actionable_insights: dict[str, Any] = field(default_factory=dict)


def _date_range(data: pd.DataFrame) -> dict[str, str]:
    return {
        "start": data[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "end": data[DATE_COLUMN].max().strftime("%Y-%m-%d"),
    }


def _within_dataset_summary(result: TrainingResult) -> dict[str, Any]:
    complete = pd.concat(
        [result.splits.train, result.splits.validation, result.splits.test],
        ignore_index=True,
    )
    positives = int(complete[TARGET_COLUMN].sum())
    return {
        "snapshots": len(complete),
        "repositories": sorted(complete[REPOSITORY_COLUMN].unique().tolist()),
        "repository_count": int(complete[REPOSITORY_COLUMN].nunique()),
        "unique_java_files": len(
            complete[[REPOSITORY_COLUMN, PATH_COLUMN]].drop_duplicates()
        ),
        "positive_labels": positives,
        "negative_labels": len(complete) - positives,
        "positive_rate": positives / len(complete),
        "date_range": _date_range(complete),
        "feature_columns": list(result.feature_columns),
        "splits": {
            name: asdict(summary)
            for name, summary in result.splits.summaries().items()
        },
    }


def _within_comparison(result: TrainingResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, model in result.models.items():
        metrics = model.test_selected
        rows.append(
            {
                "model": name,
                "selected_model": name == result.best_model_name,
                "threshold": model.selected_threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "positive_prediction_rate": metrics["positive_prediction_rate"],
            }
        )
    return pd.DataFrame(rows)


def _within_importance(result: TrainingResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, model in result.models.items():
        for record in model.feature_importance.get("ranked_features", []):
            rows.append({"model": name, **record, "fold_count": 1})
    return pd.DataFrame(rows)


def _within_metrics(result: TrainingResult) -> dict[str, Any]:
    return {
        "sentinel_version": __version__,
        "models": {
            name: {
                "selected_threshold": model.selected_threshold,
                "validation": {
                    "default_0_5": model.validation_default,
                    "selected_threshold": model.validation_selected,
                },
                "test": {
                    "default_0_5": model.test_default,
                    "selected_threshold": model.test_selected,
                },
            }
            for name, model in result.models.items()
        },
        "random_state": result.random_state,
        "leakage_controls": {
            "split": "chronological_distinct_snapshot_dates_70_15_15",
            "preprocessing": "training_only",
            "threshold_selection": "validation_only",
            "model_selection": "validation_only",
            "test_use": "final_evaluation_only",
            "fallback_explainability": "post_selection_validation_partition",
        },
    }


def _display_metric(value: Any) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.4f}"


def evaluate_within_project(
    data: pd.DataFrame,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    top_risk: int = DEFAULT_TOP_RISK,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    explain_top_k: int | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationReport:
    """Run V3 temporal evaluation under the unified V6 report contract."""
    result = train_baselines(
        chronological_split(data),
        random_state=random_state,
        progress=progress,
    )
    comparison = _within_comparison(result)
    importance = _within_importance(result)
    best = result.best_model
    best_metrics = best.test_selected
    best_importance = importance.loc[
        importance["model"] == result.best_model_name
    ].sort_values("rank")
    top_feature = (
        str(best_importance.iloc[0]["feature"])
        if not best_importance.empty
        else "unavailable"
    )
    findings = (
        f"{result.best_model_name} was selected using validation PR-AUC.",
        (
            "Its held-out test metrics were "
            f"accuracy={_display_metric(best_metrics['accuracy'])}, "
            f"F1={_display_metric(best_metrics['f1'])}, and "
            f"ROC-AUC={_display_metric(best_metrics['roc_auc'])}."
        ),
        f"Its highest-ranked explanatory feature was {top_feature}.",
    )
    risk_ranking = build_risk_ranking(
        result.splits.test,
        best.test_scores,
        model=result.best_model_name,
        evaluation_mode="within_project_temporal_test",
        score_method=best.score_method,
        decision_threshold=best.selected_threshold,
    )
    full_explanations = explain_predictions(
        best.pipeline,
        result.splits.test,
        result.feature_columns,
        best.test_scores,
        model=result.best_model_name,
        evaluation_mode="within_project_temporal_test",
        decision_threshold=best.selected_threshold,
        reference_features=result.splits.test,
    )
    risk_ranking = add_top_explanations_to_ranking(
        risk_ranking, full_explanations
    )
    prediction_explanations = sort_explanations(
        limit_explanations(full_explanations, explain_top_k)
    )
    explanation_summary = build_explanation_summary(
        prediction_explanations, explain_top_k=explain_top_k
    )
    actionable_insights = build_actionable_insights(full_explanations)
    risk_ranking = add_actionable_insights_to_ranking(
        risk_ranking, actionable_insights
    )
    risk_summary = build_risk_summary(
        risk_ranking,
        top_risk=top_risk,
        risk_threshold=risk_threshold,
        selected_model=result.best_model_name,
        evaluation_mode="within_project_temporal_test",
    )
    return EvaluationReport(
        experiment_type="within_project",
        dataset_summary=_within_dataset_summary(result),
        selected_model=result.best_model_name,
        selection_metric="validation_pr_auc",
        metrics=_within_metrics(result),
        model_comparison=comparison,
        feature_importance=importance,
        confusion_matrix=np.asarray(
            best_metrics["confusion_matrix"], dtype=int
        ),
        confusion_matrix_label=f"{result.best_model_name} on temporal test split",
        key_findings=findings,
        risk_ranking=risk_ranking,
        risk_summary=risk_summary,
        prediction_explanations=prediction_explanations,
        explanation_summary=explanation_summary,
        actionable_insights=actionable_insights,
    )


def _cross_dataset_summary(dataset: MultiRepositoryDataset) -> dict[str, Any]:
    positives = int(dataset.data[TARGET_COLUMN].sum())
    return {
        "snapshots": len(dataset.data),
        "repository_count": len(dataset.summaries),
        "repositories": [summary.repository for summary in dataset.summaries],
        "unique_repository_file_pairs": len(
            dataset.data[[REPOSITORY_COLUMN, PATH_COLUMN]].drop_duplicates()
        ),
        "positive_labels": positives,
        "negative_labels": len(dataset.data) - positives,
        "positive_rate": positives / len(dataset.data),
        "date_range": _date_range(dataset.data),
        "feature_columns": list(dataset.feature_columns),
        "repository_summaries": [
            summary.to_dict() for summary in dataset.summaries
        ],
    }


def _aggregate_cross_importance(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model, feature, method), values in raw.groupby(
        ["model", "feature", "method"], sort=False
    ):
        importance = values["importance"].astype(float).to_numpy()
        effect = pd.to_numeric(values["effect"], errors="coerce").dropna()
        directions = set(values["direction"])
        if directions == {"positive"}:
            direction = "consistently_positive"
        elif directions == {"negative"}:
            direction = "consistently_negative"
        elif directions <= {"positive", "negative", "zero"}:
            direction = "mixed"
        else:
            direction = "not_applicable"
        rows.append(
            {
                "model": model,
                "feature": feature,
                "importance": float(np.mean(importance)),
                "effect": float(effect.mean()) if not effect.empty else None,
                "direction": direction,
                "method": method,
                "standard_deviation": float(np.std(importance, ddof=0)),
                "fold_count": len(values),
            }
        )
    aggregated = pd.DataFrame(rows)
    aggregated = aggregated.sort_values(
        ["model", "importance", "feature"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    aggregated["rank"] = aggregated.groupby("model").cumcount() + 1
    return aggregated


def evaluate_cross_project(
    dataset: MultiRepositoryDataset,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    same_project_metadata: str | Path | None = None,
    top_risk: int = DEFAULT_TOP_RISK,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    explain_top_k: int | None = None,
    progress: ProgressCallback | None = None,
) -> EvaluationReport:
    """Run V4 leave-one-project-out evaluation under the V6 report contract."""
    result = run_cross_project_evaluation(
        dataset,
        random_state=random_state,
        same_project_metadata=same_project_metadata,
        progress=progress,
        retain_models=True,
    )
    comparison = result.aggregate.copy()
    selected = result.folds.loc[
        (result.folds["threshold_strategy"] == "validation_selected")
        & result.folds["is_selected_model"]
    ]
    selected_counts = selected["model"].value_counts()
    comparison["selected_fold_count"] = comparison["model"].map(
        selected_counts
    ).fillna(0).astype(int)
    matrix = np.array(
        [
            [selected["confusion_tn"].sum(), selected["confusion_fp"].sum()],
            [selected["confusion_fn"].sum(), selected["confusion_tp"].sum()],
        ],
        dtype=int,
    )
    importance = _aggregate_cross_importance(result.model_feature_importance)
    best_row = comparison.loc[comparison["mean_pr_auc"].idxmax()]
    best_model = str(best_row["model"])
    best_importance = importance.loc[importance["model"] == best_model]
    top_feature = (
        str(best_importance.sort_values("rank").iloc[0]["feature"])
        if not best_importance.empty
        else "unavailable"
    )
    findings = (
        (
            f"{best_model} had the highest mean held-out PR-AUC "
            f"({_display_metric(best_row['mean_pr_auc'])})."
        ),
        (
            "Per-fold validation selected "
            + ", ".join(
                f"{name} ({count})" for name, count in selected_counts.items()
            )
            + "."
        ),
        f"The highest-ranked aggregate {best_model} feature was {top_feature}.",
        f"The cross-project leakage audit {result.report['leakage_audit']['status']}.",
    )
    metrics = {
        "sentinel_version": __version__,
        "random_state": random_state,
        "validation_strategy": result.report["validation_strategy"],
        "per_fold_metrics": result.report["per_fold_metrics"],
        "aggregate_metrics": result.report["aggregate_metrics"],
        "model_selection": result.model_selection,
        "leakage_audit": result.report["leakage_audit"],
    }
    risk_samples: list[pd.DataFrame] = []
    risk_scores: list[np.ndarray] = []
    risk_models: list[str] = []
    risk_methods: list[str] = []
    risk_thresholds: list[float] = []
    explanation_frames: list[pd.DataFrame] = []
    for fold_evaluation in result.fold_evaluations:
        training_result = fold_evaluation.training_result
        # Defensive guard for the retained-model contract.
        if training_result is None:
            raise ValueError("Cross-project risk scoring requires retained models.")
        model = training_result.best_model
        row_count = len(fold_evaluation.fold.test)
        risk_samples.append(fold_evaluation.fold.test)
        risk_scores.append(model.test_scores)
        risk_models.extend([training_result.best_model_name] * row_count)
        risk_methods.extend([model.score_method] * row_count)
        risk_thresholds.extend([model.selected_threshold] * row_count)
        explanation_frames.append(
            explain_predictions(
                model.pipeline,
                fold_evaluation.fold.test,
                training_result.feature_columns,
                model.test_scores,
                model=training_result.best_model_name,
                evaluation_mode="cross_project_held_out_folds",
                decision_threshold=model.selected_threshold,
                reference_features=fold_evaluation.fold.test,
            )
        )
    combined_samples = pd.concat(risk_samples, ignore_index=True)
    risk_ranking = build_risk_ranking(
        combined_samples,
        np.concatenate(risk_scores),
        model=risk_models,
        evaluation_mode="cross_project_held_out_folds",
        score_method=risk_methods,
        decision_threshold=risk_thresholds,
    )
    full_explanations = pd.concat(explanation_frames, ignore_index=True)
    risk_ranking = add_top_explanations_to_ranking(
        risk_ranking, full_explanations
    )
    prediction_explanations = sort_explanations(
        limit_explanations(full_explanations, explain_top_k)
    )
    explanation_summary = build_explanation_summary(
        prediction_explanations, explain_top_k=explain_top_k
    )
    actionable_insights = build_actionable_insights(full_explanations)
    risk_ranking = add_actionable_insights_to_ranking(
        risk_ranking, actionable_insights
    )
    risk_summary = build_risk_summary(
        risk_ranking,
        top_risk=top_risk,
        risk_threshold=risk_threshold,
        selected_model="per_fold_validation_selection",
        evaluation_mode="cross_project_held_out_folds",
    )
    return EvaluationReport(
        experiment_type="cross_project",
        dataset_summary=_cross_dataset_summary(dataset),
        selected_model="per_fold_validation_selection",
        selection_metric="validation_pr_auc",
        metrics=metrics,
        model_comparison=comparison,
        feature_importance=importance,
        confusion_matrix=matrix,
        confusion_matrix_label="pooled per-fold selected models",
        key_findings=findings,
        risk_ranking=risk_ranking,
        risk_summary=risk_summary,
        prediction_explanations=prediction_explanations,
        explanation_summary=explanation_summary,
        actionable_insights=actionable_insights,
    )
