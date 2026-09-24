"""Unified Sentinel evaluation and risk report generation."""

from sentinel.evaluation.calibration import (
    CALIBRATION_COLUMNS,
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_ANALYSIS_THRESHOLDS,
    DEFAULT_CALIBRATION_BINS,
    THRESHOLD_ANALYSIS_COLUMNS,
    RiskCalibrationResult,
    build_risk_calibration,
    calibration_markdown_lines,
)
from sentinel.evaluation.action_evaluation import (
    ACTION_EVALUATION_SCHEMA_VERSION,
    ACTION_QUALITY_COLUMNS,
    ActionEvaluationResult,
    build_action_evaluation,
    build_action_evaluation_markdown,
)
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
from sentinel.evaluation.reporting import save_evaluation_report, save_risk_calibration
from sentinel.evaluation.risk import (
    build_risk_ranking,
    build_risk_summary,
    calculate_risk_scores,
)
from sentinel.ml.evaluation import calculate_metrics, extract_feature_importance

__all__ = [
    "CALIBRATION_COLUMNS",
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_ANALYSIS_THRESHOLDS",
    "DEFAULT_CALIBRATION_BINS",
    "THRESHOLD_ANALYSIS_COLUMNS",
    "RiskCalibrationResult",
    "build_risk_calibration",
    "calibration_markdown_lines",
    "save_risk_calibration",
    "ACTION_EVALUATION_SCHEMA_VERSION",
    "ACTION_QUALITY_COLUMNS",
    "ActionEvaluationResult",
    "build_action_evaluation",
    "build_action_evaluation_markdown",
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
