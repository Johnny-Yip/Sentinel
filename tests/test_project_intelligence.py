from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sentinel.evaluation.explain import EXPLANATION_COLUMNS
from sentinel.evaluation.insights import (
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.evaluation.project_intelligence import (
    DEVELOPER_PRIORITY_COLUMNS,
    PROJECT_INTELLIGENCE_SCHEMA_VERSION,
    build_project_intelligence,
    calculate_risk_concentration,
)
from sentinel.evaluation.risk import build_risk_ranking


def _explanation_row(
    *,
    project: str,
    path: str,
    date: str,
    score: float,
    feature: str,
    contribution: float,
    feature_rank: int,
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
        "normalized_contribution": abs(contribution),
        "direction": (
            "increases_risk"
            if contribution > 0.0
            else "decreases_risk"
            if contribution < 0.0
            else "neutral"
        ),
        "feature_rank": feature_rank,
    }


def _project_inputs(
    project: str = "owner/example",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    observations = [
        ("src/A.java", "2024-01-31", 0.90, "commit_count", 0.40),
        ("src/A.java", "2024-02-29", 0.80, "commit_count", 0.30),
        ("src/B.java", "2024-01-31", 0.95, "code_churn", 0.50),
        ("src/B.java", "2024-02-29", 0.70, "code_churn", 0.25),
        ("src/B.java", "2024-03-31", 0.10, "code_churn", -0.20),
        ("src/C.java", "2024-01-31", 0.60, "developer_count", -0.10),
    ]
    samples = pd.DataFrame(
        {
            "repository": [project] * len(observations),
            "file_path": [row[0] for row in observations],
            "snapshot_date": [row[1] for row in observations],
            "defect_next_90_days": [0] * len(observations),
        }
    )
    scores = [row[2] for row in observations]
    ranking = build_risk_ranking(
        samples,
        scores,
        model="example",
        evaluation_mode="within_project_temporal_test",
        score_method="predict_proba",
        decision_threshold=0.5,
    )
    rows: list[dict[str, object]] = []
    for path, date, score, feature, contribution in observations:
        rows.append(
            _explanation_row(
                project=project,
                path=path,
                date=date,
                score=score,
                feature=feature,
                contribution=contribution,
                feature_rank=1,
            )
        )
        rows.append(
            _explanation_row(
                project=project,
                path=path,
                date=date,
                score=score,
                feature="days_since_last_change",
                contribution=-0.05,
                feature_rank=2,
            )
        )
    explanations = pd.DataFrame(rows, columns=EXPLANATION_COLUMNS)
    actionable = build_actionable_insights(explanations)
    ranking = add_actionable_insights_to_ranking(ranking, actionable)
    return ranking, explanations, actionable


def test_concentration_definitions_cover_empty_small_and_zero_risk() -> None:
    assert calculate_risk_concentration([]) == {
        "aggregate_predicted_risk": 0.0,
        "top_10_percent_sample_count": 0,
        "top_10_percent_risk_contribution_percentage": 0.0,
        "top_20_percent_sample_count": 0,
        "top_20_percent_risk_contribution_percentage": 0.0,
        "samples_responsible_for_50_percent_of_predicted_risk": 0,
    }
    zero = calculate_risk_concentration([0.0])
    assert zero["top_10_percent_sample_count"] == 1
    assert zero["top_10_percent_risk_contribution_percentage"] == 0.0
    assert zero["samples_responsible_for_50_percent_of_predicted_risk"] == 0
    small = calculate_risk_concentration([0.4, 0.3, 0.2, 0.1])
    assert small["top_10_percent_sample_count"] == 1
    assert small["top_20_percent_sample_count"] == 1
    assert np.isclose(
        small["top_10_percent_risk_contribution_percentage"], 40.0
    )
    assert small["samples_responsible_for_50_percent_of_predicted_risk"] == 2


def test_aggregation_hotspots_signals_priority_and_temporal_are_deterministic() -> None:
    ranking, explanations, actionable = _project_inputs()

    first = build_project_intelligence(
        ranking,
        explanations.sample(frac=1.0, random_state=3),
        actionable,
        risk_threshold=0.5,
        experiment_type="within_project",
    )
    second = build_project_intelligence(
        ranking,
        explanations.sample(frac=1.0, random_state=9),
        actionable,
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    assert json.dumps(first.artifact, sort_keys=True) == json.dumps(
        second.artifact, sort_keys=True
    )
    pd.testing.assert_frame_equal(first.developer_priority, second.developer_priority)
    artifact = first.artifact
    assert artifact["schema_version"] == PROJECT_INTELLIGENCE_SCHEMA_VERSION
    assert artifact["summary"]["total_evaluated_samples"] == 6
    assert artifact["summary"]["high_risk_sample_count"] == 5
    assert artifact["summary"]["unique_risky_entities"] == 3
    assert [hotspot["identifier"] for hotspot in artifact["hotspots"]] == [
        "src/A.java",
        "src/B.java",
    ]
    assert artifact["hotspots"][0]["recommended_inspection_actions"]
    signal_profile = artifact["risk_signal_profile"]
    assert signal_profile["most_common_risk_signals"][0]["feature"] in {
        "code_churn",
        "commit_count",
    }
    assert signal_profile["most_common_protective_signals"][0]["feature"] == (
        "days_since_last_change"
    )
    assert signal_profile["strongest_aggregate_risk_contributors"][0][
        "feature"
    ] == "code_churn"
    assert artifact["temporal_analysis"]["supported"] is True
    assert artifact["temporal_analysis"]["direction"] == "decreasing"
    assert artifact["temporal_analysis"]["periods"][1][
        "change_in_mean_risk_from_previous_period"
    ] < 0.0

    queue = first.developer_priority
    assert list(queue.columns) == list(DEVELOPER_PRIORITY_COLUMNS)
    assert queue.iloc[0]["identifier"] == "src/A.java"
    assert np.isclose(queue.iloc[0]["priority_score"], 0.80 * 0.90 + 0.20)
    assert artifact["developer_priority"]["priority_score_is_probability"] is False


def test_missing_identifiers_and_timestamps_are_explicitly_unsupported() -> None:
    ranking, explanations, actionable = _project_inputs()
    ranking["file_path"] = ""
    ranking["snapshot_date"] = ""
    explanations["file_path"] = ""
    explanations["snapshot_date"] = ""

    result = build_project_intelligence(
        ranking,
        explanations,
        actionable,
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    assert result.artifact["hotspots"] == []
    assert result.artifact["temporal_analysis"]["supported"] is False
    assert result.artifact["metadata"]["hotspot_identifiers_available"] is False
    assert result.artifact["metadata"]["temporal_analysis_available"] is False
    assert set(result.artifact["metadata"]["unsupported_analyses"]) >= {
        "recurring_hotspots",
        "temporal_analysis",
    }
    assert result.developer_priority["identifier"].eq("").all()


def test_empty_dataset_has_stable_machine_readable_outputs() -> None:
    ranking = pd.DataFrame(
        columns=["risk_score", "project", "file_path", "snapshot_date"]
    )
    result = build_project_intelligence(
        ranking,
        pd.DataFrame(columns=EXPLANATION_COLUMNS),
        {},
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    assert result.artifact["summary"]["total_evaluated_samples"] == 0
    assert result.artifact["summary"]["mean_predicted_risk"] is None
    assert result.artifact["summary"]["risk_score_distribution"][0][
        "percentage_of_samples"
    ] == 0.0
    assert result.artifact["risk_concentration"][
        "samples_responsible_for_50_percent_of_predicted_risk"
    ] == 0
    assert result.artifact["hotspots"] == []
    assert result.developer_priority.empty
    assert list(result.developer_priority.columns) == list(
        DEVELOPER_PRIORITY_COLUMNS
    )


def test_unknown_future_features_are_reported_and_skipped() -> None:
    ranking, explanations, _ = _project_inputs()
    unknown = explanations.iloc[[0]].copy()
    unknown["feature"] = "future_complexity_index"
    unknown["contribution"] = 4.0
    unknown["absolute_contribution"] = 4.0
    unknown["direction"] = "increases_risk"
    explanations = pd.concat([explanations, unknown], ignore_index=True)
    actionable = build_actionable_insights(explanations)

    result = build_project_intelligence(
        ranking,
        explanations,
        actionable,
        risk_threshold=0.5,
        experiment_type="within_project",
    )

    assert result.artifact["metadata"]["unknown_or_unsupported_features"] == [
        "future_complexity_index"
    ]
    profile_features = {
        row["feature"]
        for row in result.artifact["risk_signal_profile"][
            "strongest_aggregate_risk_contributors"
        ]
    }
    assert "future_complexity_index" not in profile_features


def test_cross_project_intelligence_never_pools_project_risk_summaries() -> None:
    alpha = _project_inputs("owner/alpha")
    beta = _project_inputs("owner/beta")
    ranking = pd.concat([alpha[0], beta[0]], ignore_index=True)
    explanations = pd.concat([alpha[1], beta[1]], ignore_index=True)
    actionable = build_actionable_insights(explanations)

    result = build_project_intelligence(
        ranking,
        explanations,
        actionable,
        risk_threshold=0.5,
        experiment_type="cross_project",
    )

    artifact = result.artifact
    assert artifact["summary"]["analysis_scope"] == "per_project"
    assert "mean_predicted_risk" not in artifact["summary"]
    assert artifact["risk_concentration"]["available_in"] == (
        "projects[].risk_concentration"
    )
    assert [project["project"] for project in artifact["projects"]] == [
        "owner/alpha",
        "owner/beta",
    ]
    assert all(
        project["summary"]["total_evaluated_samples"] == 6
        for project in artifact["projects"]
    )
    assert result.developer_priority.groupby("project")["rank"].min().eq(1).all()

