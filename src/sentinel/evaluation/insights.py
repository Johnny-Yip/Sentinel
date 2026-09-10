"""Deterministic developer-facing insights derived from local explanations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


ACTIONABLE_INSIGHTS_SCHEMA_VERSION = 1
DEFAULT_MAX_INSIGHTS_PER_SAMPLE = 3
RANKING_INSIGHT_COLUMNS = (
    "primary_risk_reason",
    "recommended_action",
    "risk_signal_count",
)
ACTIONABLE_INSIGHT_FIELDS = (
    "feature",
    "category",
    "signal_type",
    "interpretation",
    "recommended_action",
    "feature_value",
    "contribution",
    "normalized_contribution",
    "feature_rank",
)
ACTIONABLE_INSIGHT_SAMPLE_FIELDS = (
    "sample_id",
    "project",
    "file_path",
    "snapshot_date",
    "risk_score",
    "predicted_label",
    "true_label",
    "model",
    "evaluation_mode",
    "primary_risk_reason",
    "recommended_action",
    "risk_signal_count",
    "protective_signal_count",
    "insights",
)

_REQUIRED_EXPLANATION_COLUMNS = {
    "sample_id",
    "project",
    "file_path",
    "snapshot_date",
    "risk_score",
    "predicted_label",
    "true_label",
    "model",
    "evaluation_mode",
    "feature",
    "feature_value",
    "contribution",
    "absolute_contribution",
    "normalized_contribution",
    "direction",
    "feature_rank",
}
_SAMPLE_KEY_COLUMNS = ("sample_id", "model", "evaluation_mode")
_RANKING_KEY_COLUMNS = (
    "project",
    "file_path",
    "snapshot_date",
    "model",
    "evaluation_mode",
)


@dataclass(frozen=True)
class FeatureInsightDefinition:
    """Stable domain language for one supported historical predictor."""

    category: str
    interpretation: str
    positive_contribution_signal: str
    risk_interpretation: str
    protective_interpretation: str
    risk_recommendation: str
    protective_recommendation: str


FEATURE_INSIGHTS: dict[str, FeatureInsightDefinition] = {
    "commit_count": FeatureInsightDefinition(
        category="frequent_modification",
        interpretation="How frequently the file has been modified.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "The file's modification frequency is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "The file's modification frequency is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Inspect recent edits for repeated rework, overlapping changes, and "
            "incomplete fixes."
        ),
        protective_recommendation=(
            "Confirm that the stable change pattern is supported by current tests."
        ),
    ),
    "developer_count": FeatureInsightDefinition(
        category="ownership_dispersion",
        interpretation="How many distinct developers have changed the file.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "The file's contributor and ownership pattern is pushing predicted "
            "risk upward."
        ),
        protective_interpretation=(
            "The file's contributor and ownership pattern is acting as a "
            "stabilizing signal."
        ),
        risk_recommendation=(
            "Check ownership handoffs, reviewer coverage, and assumptions shared "
            "across contributors."
        ),
        protective_recommendation=(
            "Confirm that ownership and reviewer coverage remain clear."
        ),
    ),
    "lines_added": FeatureInsightDefinition(
        category="file_growth",
        interpretation="The cumulative volume of lines added to the file.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "Historical file growth is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "Historical file growth is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Review recently added logic for concentrated complexity and missing "
            "boundary tests."
        ),
        protective_recommendation=(
            "Confirm that recent additions remain covered by focused tests."
        ),
    ),
    "lines_deleted": FeatureInsightDefinition(
        category="change_complexity",
        interpretation="The cumulative volume of lines deleted from the file.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "Historical deletion and refactoring volume is pushing predicted risk "
            "upward."
        ),
        protective_interpretation=(
            "Historical deletion and refactoring volume is acting as a stabilizing "
            "signal."
        ),
        risk_recommendation=(
            "Inspect removals and refactors for lost behavior, stale callers, and "
            "regression-test gaps."
        ),
        protective_recommendation=(
            "Confirm that simplified paths and removed behavior stay regression-tested."
        ),
    ),
    "code_churn": FeatureInsightDefinition(
        category="high_churn",
        interpretation="The cumulative total of lines added and deleted.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "Overall code churn is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "Overall code churn is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Inspect churn hotspots for repeated rewrites, broad diffs, and weak "
            "regression coverage."
        ),
        protective_recommendation=(
            "Confirm that the stable churn pattern still reflects current code paths."
        ),
    ),
    "file_age_days": FeatureInsightDefinition(
        category="file_maturity",
        interpretation="How long the file has existed in the observed history.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "The file's age and maturity pattern is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "The file's age and maturity pattern is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Review legacy assumptions, dependency boundaries, and long-standing "
            "test gaps."
        ),
        protective_recommendation=(
            "Confirm that mature behavior and compatibility assumptions remain tested."
        ),
    ),
    "days_since_last_change": FeatureInsightDefinition(
        category="change_recency",
        interpretation="How many days have passed since the latest observed change.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "The file's change-recency pattern is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "The file's change-recency pattern is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Inspect recent change clusters and dormant-code assumptions before the "
            "next edit."
        ),
        protective_recommendation=(
            "Confirm that the apparent recency stability is not masking stale tests."
        ),
    ),
    "previous_bug_fixes": FeatureInsightDefinition(
        category="defect_history",
        interpretation="How many earlier heuristic bug-fix commits touched the file.",
        positive_contribution_signal="risk",
        risk_interpretation=(
            "Prior defect-fix history is pushing predicted risk upward."
        ),
        protective_interpretation=(
            "Prior defect-fix history is acting as a stabilizing signal."
        ),
        risk_recommendation=(
            "Review recurring defect themes and strengthen regression tests around "
            "previous fixes."
        ),
        protective_recommendation=(
            "Confirm that prior fixes remain protected by regression tests."
        ),
    ),
}


def _empty_insights(max_insights_per_sample: int) -> dict[str, Any]:
    return {
        "schema_version": ACTIONABLE_INSIGHTS_SCHEMA_VERSION,
        "total_samples": 0,
        "samples_with_risk_signals": 0,
        "samples_with_protective_signals": 0,
        "total_risk_signals": 0,
        "total_protective_signals": 0,
        "common_risk_signals": [],
        "common_protective_signals": [],
        "samples": [],
        "feature_definitions": _feature_definitions(),
        "insight_metadata": {
            "deterministic": True,
            "max_insights_per_sample": max_insights_per_sample,
            "selection": "absolute_contribution_desc_then_feature_name",
            "signal_semantics": (
                "positive_contributions_are_risk_signals_and_negative_"
                "contributions_are_protective_signals"
            ),
            "supported_features": list(FEATURE_INSIGHTS),
            "unsupported_features": [],
        },
    }


def _feature_definitions() -> list[dict[str, str]]:
    return [
        {"feature": feature, **asdict(definition)}
        for feature, definition in FEATURE_INSIGHTS.items()
    ]


def _optional_int(value: Any) -> int | None:
    return None if pd.isna(value) else int(value)


def _sample_insight(row: Any, definition: FeatureInsightDefinition) -> dict[str, Any]:
    signal_type = "risk" if row.direction == "increases_risk" else "protective"
    if signal_type == "risk":
        interpretation = definition.risk_interpretation
        recommendation = definition.risk_recommendation
    else:
        interpretation = definition.protective_interpretation
        recommendation = definition.protective_recommendation
    return {
        "feature": str(row.feature),
        "category": definition.category,
        "signal_type": signal_type,
        "interpretation": interpretation,
        "recommended_action": recommendation,
        "feature_value": float(row.feature_value),
        "contribution": float(row.contribution),
        "normalized_contribution": float(row.normalized_contribution),
        "feature_rank": int(row.feature_rank),
    }


def _common_signals(
    counts: dict[str, int], signal_type: str
) -> list[dict[str, Any]]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "feature": feature,
            "category": FEATURE_INSIGHTS[feature].category,
            "interpretation": (
                FEATURE_INSIGHTS[feature].risk_interpretation
                if signal_type == "risk"
                else FEATURE_INSIGHTS[feature].protective_interpretation
            ),
            "sample_count": count,
        }
        for feature, count in ordered[:10]
    ]


def build_actionable_insights(
    explanations: pd.DataFrame,
    *,
    max_insights_per_sample: int = DEFAULT_MAX_INSIGHTS_PER_SAMPLE,
) -> dict[str, Any]:
    """Map Phase 2 contributions to deterministic developer-facing guidance."""
    if max_insights_per_sample <= 0:
        raise ValueError("max_insights_per_sample must be positive.")
    if explanations.empty:
        return _empty_insights(max_insights_per_sample)

    missing = sorted(_REQUIRED_EXPLANATION_COLUMNS - set(explanations.columns))
    if missing:
        raise ValueError(
            "Actionable insights require explanation columns: " + ", ".join(missing)
        )
    directions = set(explanations["direction"].unique())
    invalid_directions = sorted(
        directions - {"increases_risk", "decreases_risk", "neutral"}
    )
    if invalid_directions:
        raise ValueError(
            "Unsupported explanation directions: " + ", ".join(invalid_directions)
        )
    expected_directions = explanations["contribution"].map(
        lambda value: (
            "increases_risk"
            if float(value) > 0.0
            else "decreases_risk"
            if float(value) < 0.0
            else "neutral"
        )
    )
    if not expected_directions.equals(explanations["direction"]):
        raise ValueError(
            "Explanation directions must match their contribution signs."
        )

    ordered = explanations.sort_values(
        [
            "risk_score",
            "project",
            "file_path",
            "snapshot_date",
            "model",
            "feature_rank",
            "feature",
        ],
        ascending=[False, True, True, True, True, True, True],
        kind="stable",
    )
    samples: list[dict[str, Any]] = []
    total_risk_signals = 0
    total_protective_signals = 0
    risk_feature_counts: dict[str, int] = {}
    protective_feature_counts: dict[str, int] = {}
    unsupported_features = sorted(
        set(ordered["feature"].astype(str)) - set(FEATURE_INSIGHTS)
    )
    for _, group in ordered.groupby(list(_SAMPLE_KEY_COLUMNS), sort=False):
        first = group.iloc[0]
        supported = group.loc[group["feature"].isin(FEATURE_INSIGHTS)].copy()
        risk_rows = supported.loc[supported["direction"] == "increases_risk"]
        protective_rows = supported.loc[
            supported["direction"] == "decreases_risk"
        ]
        risk_signal_count = len(risk_rows)
        protective_signal_count = len(protective_rows)
        total_risk_signals += risk_signal_count
        total_protective_signals += protective_signal_count
        for feature in risk_rows["feature"].astype(str).unique():
            risk_feature_counts[feature] = risk_feature_counts.get(feature, 0) + 1
        for feature in protective_rows["feature"].astype(str).unique():
            protective_feature_counts[feature] = (
                protective_feature_counts.get(feature, 0) + 1
            )

        strongest_risk = risk_rows.sort_values(
            ["absolute_contribution", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        strongest_protective = protective_rows.sort_values(
            ["absolute_contribution", "feature"],
            ascending=[False, True],
            kind="stable",
        )
        if not strongest_risk.empty:
            primary_feature = str(strongest_risk.iloc[0]["feature"])
            primary_definition = FEATURE_INSIGHTS[primary_feature]
            primary_reason = primary_definition.risk_interpretation
            recommended_action = primary_definition.risk_recommendation
        elif not strongest_protective.empty:
            primary_feature = str(strongest_protective.iloc[0]["feature"])
            primary_definition = FEATURE_INSIGHTS[primary_feature]
            primary_reason = "No material risk-increasing contribution identified."
            recommended_action = primary_definition.protective_recommendation
        else:
            primary_reason = "No non-neutral supported feature contribution identified."
            recommended_action = (
                "Review the detailed explanation before prioritizing inspection."
            )

        selected = (
            supported.loc[supported["direction"] != "neutral"]
            .sort_values(
                ["absolute_contribution", "feature", "feature_rank"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(max_insights_per_sample)
        )
        insight_rows = [
            _sample_insight(row, FEATURE_INSIGHTS[str(row.feature)])
            for row in selected.itertuples(index=False)
        ]
        sample = {
            "sample_id": str(first["sample_id"]),
            "project": str(first["project"]),
            "file_path": str(first["file_path"]),
            "snapshot_date": str(first["snapshot_date"]),
            "risk_score": float(first["risk_score"]),
            "predicted_label": int(first["predicted_label"]),
            "true_label": _optional_int(first["true_label"]),
            "model": str(first["model"]),
            "evaluation_mode": str(first["evaluation_mode"]),
            "primary_risk_reason": primary_reason,
            "recommended_action": recommended_action,
            "risk_signal_count": risk_signal_count,
            "protective_signal_count": protective_signal_count,
            "insights": insight_rows,
        }
        samples.append(sample)

    result = _empty_insights(max_insights_per_sample)
    result.update(
        {
            "total_samples": len(samples),
            "samples_with_risk_signals": sum(
                sample["risk_signal_count"] > 0 for sample in samples
            ),
            "samples_with_protective_signals": sum(
                sample["protective_signal_count"] > 0 for sample in samples
            ),
            "total_risk_signals": total_risk_signals,
            "total_protective_signals": total_protective_signals,
            "common_risk_signals": _common_signals(risk_feature_counts, "risk"),
            "common_protective_signals": _common_signals(
                protective_feature_counts, "protective"
            ),
            "samples": samples,
        }
    )
    result["insight_metadata"]["unsupported_features"] = unsupported_features
    return result


def add_actionable_insights_to_ranking(
    ranking: pd.DataFrame, actionable_insights: dict[str, Any]
) -> pd.DataFrame:
    """Add concise insight fields to the existing sample-level risk ranking."""
    by_key = {
        tuple(sample[column] for column in _RANKING_KEY_COLUMNS): sample
        for sample in actionable_insights.get("samples", [])
    }
    result = ranking.copy()
    reasons: list[str] = []
    recommendations: list[str] = []
    counts: list[int] = []
    for row in result.itertuples(index=False):
        key = tuple(getattr(row, column) for column in _RANKING_KEY_COLUMNS)
        sample = by_key.get(key)
        if sample is None:
            reasons.append("No actionable explanation was available.")
            recommendations.append("Review the detailed prediction output manually.")
            counts.append(0)
        else:
            reasons.append(str(sample["primary_risk_reason"]))
            recommendations.append(str(sample["recommended_action"]))
            counts.append(int(sample["risk_signal_count"]))
    result["primary_risk_reason"] = reasons
    result["recommended_action"] = recommendations
    result["risk_signal_count"] = counts
    return result
