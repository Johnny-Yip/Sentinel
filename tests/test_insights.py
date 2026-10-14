from __future__ import annotations

import pandas as pd
import pytest

from sentinel.evaluation.explain import EXPLANATION_COLUMNS
from sentinel.evaluation.insights import (
    ACTIONABLE_INSIGHT_FIELDS,
    ACTIONABLE_INSIGHT_SAMPLE_FIELDS,
    ACTIONABLE_INSIGHTS_SCHEMA_VERSION,
    FEATURE_INSIGHTS,
    RANKING_INSIGHT_COLUMNS,
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.ml.data import MODEL_FEATURE_COLUMNS


def explanation_row(
    *,
    sample_id: str,
    file_path: str,
    feature: str,
    contribution: float,
    feature_rank: int,
    risk_score: float = 0.8,
) -> dict[str, object]:
    direction = (
        "increases_risk"
        if contribution > 0.0
        else "decreases_risk"
        if contribution < 0.0
        else "neutral"
    )
    return {
        "sample_id": sample_id,
        "project": "owner/example",
        "file_path": file_path,
        "snapshot_date": "2024-01-31",
        "risk_score": risk_score,
        "predicted_label": int(risk_score >= 0.5),
        "true_label": 1,
        "model": "example",
        "evaluation_mode": "within_project_temporal_test",
        "explanation_method": "test",
        "contribution_space": "probability_delta",
        "feature": feature,
        "feature_value": 7.0,
        "model_input_value": 0.4,
        "contribution": contribution,
        "absolute_contribution": abs(contribution),
        "normalized_contribution": abs(contribution),
        "direction": direction,
        "feature_rank": feature_rank,
    }


def explanation_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=EXPLANATION_COLUMNS)


def test_feature_registry_covers_every_model_feature() -> None:
    assert list(FEATURE_INSIGHTS) == MODEL_FEATURE_COLUMNS
    for definition in FEATURE_INSIGHTS.values():
        assert definition.interpretation
        assert definition.positive_contribution_signal == "risk"
        assert definition.risk_recommendation
        assert definition.protective_recommendation


def test_insights_are_deterministic_and_map_strongest_contributions() -> None:
    explanations = explanation_frame(
        [
            explanation_row(
                sample_id="sample-a",
                file_path="src/A.java",
                feature="previous_bug_fixes",
                contribution=0.2,
                feature_rank=3,
            ),
            explanation_row(
                sample_id="sample-a",
                file_path="src/A.java",
                feature="developer_count",
                contribution=-0.5,
                feature_rank=2,
            ),
            explanation_row(
                sample_id="sample-a",
                file_path="src/A.java",
                feature="commit_count",
                contribution=0.6,
                feature_rank=1,
            ),
        ]
    )

    first = build_actionable_insights(explanations, max_insights_per_sample=2)
    shuffled = explanations.sample(frac=1.0, random_state=9).reset_index(drop=True)
    second = build_actionable_insights(shuffled, max_insights_per_sample=2)

    assert first == second
    sample = first["samples"][0]
    assert sample["primary_risk_reason"] == (
        FEATURE_INSIGHTS["commit_count"].risk_interpretation
    )
    assert sample["recommended_action"] == (
        FEATURE_INSIGHTS["commit_count"].risk_recommendation
    )
    assert sample["risk_signal_count"] == 2
    assert sample["protective_signal_count"] == 1
    assert [insight["feature"] for insight in sample["insights"]] == [
        "commit_count",
        "developer_count",
    ]
    assert [insight["signal_type"] for insight in sample["insights"]] == [
        "risk",
        "protective",
    ]


def test_protective_only_sample_uses_protective_recommendation() -> None:
    explanations = explanation_frame(
        [
            explanation_row(
                sample_id="sample-b",
                file_path="src/B.java",
                feature="previous_bug_fixes",
                contribution=-0.7,
                feature_rank=1,
            )
        ]
    )

    result = build_actionable_insights(explanations)

    sample = result["samples"][0]
    assert sample["risk_signal_count"] == 0
    assert sample["protective_signal_count"] == 1
    assert sample["primary_risk_reason"] == (
        "No material risk-increasing contribution identified."
    )
    assert sample["recommended_action"] == (
        FEATURE_INSIGHTS["previous_bug_fixes"].protective_recommendation
    )
    assert sample["insights"][0]["interpretation"] == (
        FEATURE_INSIGHTS["previous_bug_fixes"].protective_interpretation
    )
    assert result["common_protective_signals"][0]["sample_count"] == 1


def test_empty_and_unsupported_explanations_have_explicit_fallbacks() -> None:
    empty = build_actionable_insights(pd.DataFrame(columns=EXPLANATION_COLUMNS))
    assert empty["total_samples"] == 0
    assert empty["samples"] == []

    unknown = explanation_frame(
        [
            explanation_row(
                sample_id="sample-c",
                file_path="src/C.java",
                feature="future_feature",
                contribution=0.4,
                feature_rank=1,
            )
        ]
    )
    result = build_actionable_insights(unknown)

    assert result["samples"][0]["risk_signal_count"] == 0
    assert result["samples"][0]["insights"] == []
    assert result["insight_metadata"]["unsupported_features"] == [
        "future_feature"
    ]


def test_inconsistent_explanation_direction_is_rejected() -> None:
    explanations = explanation_frame(
        [
            explanation_row(
                sample_id="sample-d",
                file_path="src/D.java",
                feature="code_churn",
                contribution=0.4,
                feature_rank=1,
            )
        ]
    )
    explanations.loc[0, "direction"] = "decreases_risk"

    with pytest.raises(ValueError, match="contribution signs"):
        build_actionable_insights(explanations)


def test_output_schemas_and_ranking_enrichment() -> None:
    explanations = explanation_frame(
        [
            explanation_row(
                sample_id="sample-e",
                file_path="src/E.java",
                feature="code_churn",
                contribution=0.4,
                feature_rank=1,
            )
        ]
    )
    result = build_actionable_insights(explanations)
    sample = result["samples"][0]
    ranking = pd.DataFrame(
        [
            {
                "project": sample["project"],
                "file_path": sample["file_path"],
                "snapshot_date": sample["snapshot_date"],
                "model": sample["model"],
                "evaluation_mode": sample["evaluation_mode"],
            }
        ]
    )

    enriched = add_actionable_insights_to_ranking(ranking, result)

    assert result["schema_version"] == ACTIONABLE_INSIGHTS_SCHEMA_VERSION
    assert tuple(sample) == ACTIONABLE_INSIGHT_SAMPLE_FIELDS
    assert tuple(sample["insights"][0]) == ACTIONABLE_INSIGHT_FIELDS
    assert all(column in enriched for column in RANKING_INSIGHT_COLUMNS)
    assert enriched.loc[0, "primary_risk_reason"] == (
        FEATURE_INSIGHTS["code_churn"].risk_interpretation
    )
    assert enriched.loc[0, "risk_signal_count"] == 1
