"""Unified Sentinel V5 evaluation and report generation."""

from sentinel.evaluation.experiment import (
    EvaluationReport,
    evaluate_cross_project,
    evaluate_within_project,
)
from sentinel.evaluation.reporting import save_evaluation_report
from sentinel.ml.evaluation import calculate_metrics, extract_feature_importance

__all__ = [
    "EvaluationReport",
    "calculate_metrics",
    "evaluate_cross_project",
    "evaluate_within_project",
    "extract_feature_importance",
    "save_evaluation_report",
]
