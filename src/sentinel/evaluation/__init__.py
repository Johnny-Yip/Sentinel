"""Unified Sentinel evaluation and risk report generation."""

from sentinel.evaluation.explain import (
    add_top_explanations_to_ranking,
    build_explanation_summary,
    explain_predictions,
)
from sentinel.evaluation.experiment import (
    EvaluationReport,
    evaluate_cross_project,
    evaluate_within_project,
)
from sentinel.evaluation.reporting import save_evaluation_report
from sentinel.evaluation.risk import (
    build_risk_ranking,
    build_risk_summary,
    calculate_risk_scores,
)
from sentinel.ml.evaluation import calculate_metrics, extract_feature_importance

__all__ = [
    "EvaluationReport",
    "explain_predictions",
    "build_explanation_summary",
    "add_top_explanations_to_ranking",
    "calculate_metrics",
    "calculate_risk_scores",
    "build_risk_ranking",
    "build_risk_summary",
    "evaluate_cross_project",
    "evaluate_within_project",
    "extract_feature_importance",
    "save_evaluation_report",
]
