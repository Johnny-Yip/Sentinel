from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.decision_brief import (
    DECISION_BRIEF_SCHEMA_VERSION,
    DEVELOPER_ACTION_COLUMNS,
    assign_risk_tiers,
    attention_level,
    build_decision_brief,
    build_decision_brief_markdown,
)
from sentinel.evaluation.explain import EXPLANATION_COLUMNS
from sentinel.evaluation.insights import (
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.evaluation.project_intelligence import build_project_intelligence
from sentinel.evaluation.risk import build_risk_ranking


def _explanation(
    project: str,
    path: str,
    date: str,
    score: float,
    feature: str,
    contribution: float,
) -> dict[str, object]:
    return {
        "sample_id": f"{project}|{path}|{date}",
        "project": project,
        "file_path": path,
        "snapshot_date": date,
        "risk_score": score,
        "predicted_label": int(score >= 0.5),
        "true_label": 0,
        "model": "example",
        "evaluation_mode": "within_project_temporal_test",
        "explanation_method": "linear_coefficient_transformed",
        "contribution_space": "log_odds",
        "feature": feature,
        "feature_value": 4.0,
        "model_input_value": 0.5,
        "contribution": contribution,
        "absolute_contribution": abs(contribution),
        "normalized_contribution": 1.0,
        "direction": (
            "increases_risk"
            if contribution > 0.0
            else "decreases_risk"
            if contribution < 0.0
            else "neutral"
        ),
        "feature_rank": 1,
    }


def _project_outputs(
    project: str = "owner/example",
    *,
    paths: list[str] | None = None,
    dates: list[str] | None = None,
    scores: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], object]:
    scores = scores or [0.95, 0.85, 0.75, 0.65, 0.55, 0.40]
    paths = paths or [
        "src/A.java",
        "src/A.java",
        "src/B.java",
        "src/C.java",
        "src/D.java",
        "src/E.java",
    ]
    dates = dates or pd.date_range(
        "2024-01-31", periods=len(scores), freq="ME"
    ).strftime("%Y-%m-%d").tolist()
    samples = pd.DataFrame(
        {
            "repository": [project] * len(scores),
            "file_path": paths,
            "snapshot_date": dates,
            "defect_next_90_days": [0] * len(scores),
        }
    )
    ranking = build_risk_ranking(
        samples,
        scores,
        model="example",
        evaluation_mode="within_project_temporal_test",
        score_method="predict_proba",
        decision_threshold=0.5,
    )
    explanations = pd.DataFrame(
        [
            _explanation(
                project,
                path,
                date,
                score,
                "commit_count" if index % 2 == 0 else "code_churn",
                score / 2,
            )
            for index, (path, date, score) in enumerate(zip(paths, dates, scores))
        ],
        columns=EXPLANATION_COLUMNS,
    )
    actionable = build_actionable_insights(explanations)
    ranking = add_actionable_insights_to_ranking(ranking, actionable)
    intelligence = build_project_intelligence(
        ranking,
        explanations,
        actionable,
        risk_threshold=0.5,
        experiment_type="within_project",
    )
    return ranking, explanations, actionable, intelligence


def _priority_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": ["owner/example"] * len(scores),
            "identifier": [f"src/File{i}.java" for i in range(len(scores))],
            "snapshot_date": [f"2024-01-{i + 1:02d}" for i in range(len(scores))],
            "predicted_risk": scores,
            "priority_score": scores,
            "primary_risk_reason": ["Existing model reason."] * len(scores),
            "recommended_action": ["Inspect the relevant changes."] * len(scores),
            "risk_signal_count": [1] * len(scores),
            "recurring_high_risk_count": [0] * len(scores),
            "recurring_high_risk_rate": [0.0] * len(scores),
        }
    )


def test_risk_tiers_use_project_local_high_risk_priority_bands() -> None:
    priority = _priority_frame(
        [0.99, 0.90, 0.80, 0.70, 0.60, 0.55, 0.49, 0.30, 0.10, 0.00]
    )

    first = assign_risk_tiers(priority, risk_threshold=0.5)
    second = assign_risk_tiers(
        priority.sample(frac=1.0, random_state=7), risk_threshold=0.5
    )

    assert first[["identifier", "risk_tier"]].to_dict("records") == second[
        ["identifier", "risk_tier"]
    ].to_dict("records")
    assert first["risk_tier"].tolist() == [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "MEDIUM",
        "MEDIUM",
        "MEDIUM",
        "LOW",
        "LOW",
        "LOW",
        "LOW",
    ]
    assert first["priority_rank"].dropna().astype(int).tolist() == list(range(1, 7))


@pytest.mark.parametrize(
    ("high_risk_count", "total_samples", "expected"),
    [(0, 0, "LOW"), (0, 20, "LOW"), (1, 20, "MODERATE"), (2, 20, "ELEVATED"), (5, 20, "HIGH")],
)
def test_attention_level_boundaries(
    high_risk_count: int, total_samples: int, expected: str
) -> None:
    assert attention_level(high_risk_count, total_samples) == expected


def test_decision_brief_is_stable_when_all_equivalent_inputs_are_shuffled() -> None:
    ranking, explanations, actionable, intelligence = _project_outputs()
    shuffled_actionable = dict(actionable)
    shuffled_actionable["samples"] = list(reversed(actionable["samples"]))

    first = build_decision_brief(
        ranking,
        explanations,
        actionable,
        intelligence.artifact,
        intelligence.developer_priority,
        risk_threshold=0.5,
        experiment_type="within_project",
        model_metadata={"selected_model": "example"},
    )
    second = build_decision_brief(
        ranking.sample(frac=1.0, random_state=3),
        explanations.sample(frac=1.0, random_state=5),
        shuffled_actionable,
        intelligence.artifact,
        intelligence.developer_priority.sample(frac=1.0, random_state=9),
        risk_threshold=0.5,
        experiment_type="within_project",
        model_metadata={"selected_model": "example"},
    )

    assert json.dumps(first.artifact, sort_keys=True) == json.dumps(
        second.artifact, sort_keys=True
    )
    pd.testing.assert_frame_equal(first.developer_actions, second.developer_actions)
    summary = first.artifact["projects"][0]["executive_summary"]
    assert summary["risk_tier_counts"] == {
        "CRITICAL": 1,
        "HIGH": 1,
        "MEDIUM": 3,
        "LOW": 1,
    }
    assert summary["overall_attention_level"] == "HIGH"
    assert first.artifact["projects"][0]["top_developer_actions"][0][
        "recommended_action"
    ].startswith("Inspect")
    assert "explanation_method" in json.loads(
        first.developer_actions.iloc[0]["supporting_evidence"]
    )


def test_missing_identifier_and_timestamp_are_never_fabricated() -> None:
    ranking, explanations, actionable, intelligence = _project_outputs(
        paths=[""], dates=[""], scores=[0.9]
    )

    result = build_decision_brief(
        ranking,
        explanations,
        actionable,
        intelligence.artifact,
        intelligence.developer_priority,
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    summary = result.artifact["projects"][0]["executive_summary"]
    action = result.artifact["projects"][0]["top_developer_actions"][0]
    assert summary["top_recurring_hotspot"] is None
    assert summary["temporal_direction"] == "not_available"
    assert action["identifier"] is None
    assert action["file_path"] is None
    assert action["snapshot_date"] is None
    assert action["dominant_signal"] is None
    assert "contribution" not in action["supporting_evidence"]


def test_empty_and_zero_risk_inputs_produce_low_attention_and_no_actions() -> None:
    ranking = pd.DataFrame(
        {
            "risk_score": [0.0, 0.0],
            "project": ["owner/zero", "owner/zero"],
            "file_path": ["src/A.java", "src/B.java"],
            "snapshot_date": ["2024-01-31", "2024-02-29"],
        }
    )
    intelligence = build_project_intelligence(
        ranking,
        pd.DataFrame(columns=EXPLANATION_COLUMNS),
        {},
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    result = build_decision_brief(
        ranking,
        pd.DataFrame(columns=EXPLANATION_COLUMNS),
        {},
        intelligence.artifact,
        intelligence.developer_priority,
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    summary = result.artifact["projects"][0]["executive_summary"]
    assert result.artifact["schema_version"] == DECISION_BRIEF_SCHEMA_VERSION
    assert summary["overall_attention_level"] == "LOW"
    assert summary["risk_tier_counts"]["LOW"] == 2
    assert summary["number_of_priority_items"] == 0
    assert result.developer_actions.empty
    assert list(result.developer_actions.columns) == list(DEVELOPER_ACTION_COLUMNS)
    assert "No samples met" in build_decision_brief_markdown(result.artifact)

    empty_ranking = ranking.iloc[0:0].copy()
    empty_intelligence = build_project_intelligence(
        empty_ranking,
        pd.DataFrame(columns=EXPLANATION_COLUMNS),
        {},
        risk_threshold=0.5,
        experiment_type="within_project",
    )
    empty = build_decision_brief(
        empty_ranking,
        pd.DataFrame(columns=EXPLANATION_COLUMNS),
        {},
        empty_intelligence.artifact,
        empty_intelligence.developer_priority,
        risk_threshold=0.5,
        experiment_type="within_project",
    )
    assert empty.artifact["projects"][0]["executive_summary"]["total_samples"] == 0
    assert empty.artifact["projects"][0]["executive_summary"][
        "overall_attention_level"
    ] == "LOW"
    assert empty.developer_actions.empty


def test_cross_project_briefs_restart_ranks_and_keep_summaries_independent() -> None:
    alpha = _project_outputs("owner/alpha")
    beta = _project_outputs("owner/beta", scores=[0.9, 0.4, 0.3, 0.2, 0.1, 0.0])
    ranking = pd.concat([alpha[0], beta[0]], ignore_index=True)
    explanations = pd.concat([alpha[1], beta[1]], ignore_index=True)
    actionable = build_actionable_insights(explanations)
    ranking = add_actionable_insights_to_ranking(ranking, actionable)
    intelligence = build_project_intelligence(
        ranking,
        explanations,
        actionable,
        risk_threshold=0.5,
        experiment_type="cross_project",
    )

    result = build_decision_brief(
        ranking,
        explanations,
        actionable,
        intelligence.artifact,
        intelligence.developer_priority,
        risk_threshold=0.5,
        experiment_type="cross_project",
    )

    assert result.artifact["analysis_scope"] == "per_project"
    assert [row["project"] for row in result.artifact["projects"]] == [
        "owner/alpha",
        "owner/beta",
    ]
    summaries = {
        row["project"]: row["executive_summary"]
        for row in result.artifact["projects"]
    }
    assert summaries["owner/alpha"]["high_risk_count"] == 5
    assert summaries["owner/beta"]["high_risk_count"] == 1
    assert (
        result.developer_actions.groupby("project")["priority_rank"]
        .min()
        .eq(1)
        .all()
    )
    assert (
        result.developer_actions.groupby("project")["priority_rank"]
        .max()
        .to_dict()
    ) == {
        "owner/alpha": 5,
        "owner/beta": 1,
    }
