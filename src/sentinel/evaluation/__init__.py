"""Unified Sentinel evaluation and risk report generation."""

from sentinel.evaluation.decision_brief import (
    ATTENTION_LEVELS,
    DECISION_BRIEF_SCHEMA_VERSION,
    DEVELOPER_ACTION_COLUMNS,
    RISK_TIERS,
    DecisionBriefResult,
    assign_risk_tiers,
    attention_level,
    build_decision_brief,
    build_decision_brief_markdown,
)
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
from sentinel.evaluation.insights import (
    FEATURE_INSIGHTS,
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.evaluation.project_intelligence import (
    DEVELOPER_PRIORITY_COLUMNS,
    PROJECT_INTELLIGENCE_SCHEMA_VERSION,
    ProjectIntelligenceResult,
    build_project_intelligence,
    calculate_risk_concentration,
)
from sentinel.evaluation.reporting import save_evaluation_report
from sentinel.evaluation.risk import (
    build_risk_ranking,
    build_risk_summary,
    calculate_risk_scores,
)
from sentinel.ml.evaluation import calculate_metrics, extract_feature_importance

__all__ = [
    "ATTENTION_LEVELS",
    "DECISION_BRIEF_SCHEMA_VERSION",
    "DEVELOPER_ACTION_COLUMNS",
    "RISK_TIERS",
    "DecisionBriefResult",
    "EvaluationReport",
    "FEATURE_INSIGHTS",
    "explain_predictions",
    "build_explanation_summary",
    "add_top_explanations_to_ranking",
    "build_actionable_insights",
    "build_project_intelligence",
    "calculate_risk_concentration",
    "DEVELOPER_PRIORITY_COLUMNS",
    "PROJECT_INTELLIGENCE_SCHEMA_VERSION",
    "ProjectIntelligenceResult",
    "add_actionable_insights_to_ranking",
    "assign_risk_tiers",
    "attention_level",
    "build_decision_brief",
    "build_decision_brief_markdown",
    "calculate_metrics",
    "calculate_risk_scores",
    "build_risk_ranking",
    "build_risk_summary",
    "evaluate_cross_project",
    "evaluate_within_project",
    "extract_feature_importance",
    "save_evaluation_report",
]
