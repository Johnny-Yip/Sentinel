from __future__ import annotations

from io import StringIO
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation.action_evaluation import (
    ACTION_EVALUATION_SCHEMA_VERSION,
    ACTION_QUALITY_COLUMNS,
    build_action_evaluation,
)
from sentinel.evaluation.decision_brief import DEVELOPER_ACTION_COLUMNS
from sentinel.evaluation.explain import EXPLANATION_COLUMNS


def _ranking(
    project: str = "owner/alpha",
    *,
    scores: tuple[float, ...] = (0.9, 0.8),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": [project] * len(scores),
            "file_path": ["src/A.java"] * len(scores),
            "snapshot_date": ["2024-01-31", "2024-02-29"][: len(scores)],
            "model": ["example"] * len(scores),
            "evaluation_mode": ["within_project_temporal_test"] * len(scores),
            "risk_score": scores,
        }
    )


def _explanations(
    ranking: pd.DataFrame,
    vectors: tuple[tuple[float, float], ...] = ((0.3, -0.1), (0.6, -0.2)),
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sample, vector in zip(ranking.to_dict("records"), vectors):
        magnitude = sum(abs(value) for value in vector)
        for rank, (feature, weight) in enumerate(
            zip(("code_churn", "commit_count"), vector), start=1
        ):
            records.append(
                {
                    **sample,
                    "sample_id": (
                        f"{sample['project']}|{sample['file_path']}|"
                        f"{sample['snapshot_date']}"
                    ),
                    "predicted_label": int(sample["risk_score"] >= 0.5),
                    "true_label": 0,
                    "explanation_method": "linear_coefficient_transformed",
                    "contribution_space": "log_odds",
                    "feature": feature,
                    "feature_value": 4.0,
                    "model_input_value": 0.5,
                    "contribution": weight,
                    "absolute_contribution": abs(weight),
                    "normalized_contribution": (
                        abs(weight) / magnitude if magnitude else 0.0
                    ),
                    "direction": (
                        "increases_risk"
                        if weight > 0
                        else "decreases_risk" if weight < 0 else "neutral"
                    ),
                    "feature_rank": rank,
                }
            )
    return pd.DataFrame(records, columns=EXPLANATION_COLUMNS)


def _actions(ranking: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rank, sample in enumerate(ranking.to_dict("records"), start=1):
        records.append(
            {
                "project": sample["project"],
                "priority_rank": rank,
                "risk_tier": "CRITICAL" if rank == 1 else "MEDIUM",
                "identifier": sample["file_path"],
                "file_path": sample["file_path"],
                "snapshot_date": sample["snapshot_date"],
                "predicted_risk": sample["risk_score"],
                "priority_score": sample["risk_score"],
                "short_reason": "Overall code churn is pushing predicted risk upward.",
                "dominant_signal": "code_churn",
                "recommended_action": "Inspect recent edits for repeated rework and regression-test gaps.",
                "supporting_evidence": json.dumps(
                    {
                        "risk_score": sample["risk_score"],
                        "risk_threshold": 0.5,
                        "recurring_high_risk_count": 2,
                        "recurring_high_risk_rate": 1.0,
                        "risk_signal_count": 1,
                        "dominant_signal": "code_churn",
                        "contribution": 0.3 if rank == 1 else 0.6,
                        "explanation_method": "linear_coefficient_transformed",
                        "contribution_space": "log_odds",
                    }
                ),
            }
        )
    return pd.DataFrame(records, columns=DEVELOPER_ACTION_COLUMNS)


def _evaluate(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    actions: pd.DataFrame | None = None,
    *,
    experiment_type: str = "within_project",
):
    return build_action_evaluation(
        ranking,
        explanations,
        _actions(ranking) if actions is None else actions,
        risk_threshold=0.5,
        experiment_type=experiment_type,
    )


def test_evaluation_is_deterministic_and_preserves_all_inputs() -> None:
    ranking = _ranking()
    explanations = _explanations(ranking)
    actions = _actions(ranking)
    originals = [frame.copy(deep=True) for frame in (ranking, explanations, actions)]

    first = _evaluate(ranking, explanations, actions)
    second = _evaluate(
        ranking.sample(frac=1, random_state=1),
        explanations.sample(frac=1, random_state=2),
        actions.sample(frac=1, random_state=3),
    )

    assert json.dumps(first.artifact, sort_keys=True, allow_nan=False) == json.dumps(
        second.artifact, sort_keys=True, allow_nan=False
    )
    pd.testing.assert_frame_equal(first.action_quality, second.action_quality)
    for frame, original in zip((ranking, explanations, actions), originals):
        pd.testing.assert_frame_equal(frame, original)


def test_stability_uses_signed_retained_vectors_and_alignment_magnitudes() -> None:
    ranking = _ranking()
    result = _evaluate(ranking, _explanations(ranking))
    project = result.artifact["projects"][0]
    stability = project["explanation_stability"]
    alignment = project["risk_explanation_alignment"]

    assert stability["mean_explanation_similarity"] == pytest.approx(1.0)
    assert stability["median_explanation_similarity"] == pytest.approx(1.0)
    assert stability["stability_pair_count"] == 1
    assert stability["stability_sample_count"] == 2
    assert alignment["high_risk_sample_count"] == 2
    assert alignment["explained_high_risk_count"] == 2
    assert alignment["explanation_coverage_rate"] == 1.0
    assert alignment["mean_top_feature_share"] == pytest.approx(0.75)
    assert alignment["mean_explanation_strength"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "mismatch",
    [
        "path",
        "date_gap",
        "model",
        "evaluation_mode",
        "method",
        "space",
        "features",
        "zero",
    ],
)
def test_stability_rejects_incomparable_or_missing_vectors(mismatch: str) -> None:
    ranking = _ranking()
    if mismatch == "path":
        ranking.loc[1, "file_path"] = "src/B.java"
    elif mismatch == "date_gap":
        ranking.loc[1, "snapshot_date"] = "2024-05-31"
    elif mismatch in {"model", "evaluation_mode"}:
        ranking.loc[1, mismatch] = "different"
    explanations = _explanations(ranking)
    second_sample = explanations.index >= 2
    if mismatch == "method":
        explanations.loc[second_sample, "explanation_method"] = (
            "local_median_perturbation"
        )
    elif mismatch == "space":
        explanations.loc[second_sample, "contribution_space"] = "probability_delta"
    elif mismatch == "features":
        explanations = explanations.drop(index=3)
    elif mismatch == "zero":
        explanations.loc[second_sample, ["contribution", "absolute_contribution"]] = 0.0

    stability = _evaluate(ranking, explanations).artifact["projects"][0][
        "explanation_stability"
    ]

    assert stability["mean_explanation_similarity"] is None
    assert stability["median_explanation_similarity"] is None
    assert stability["stability_pair_count"] == 0
    assert stability["stability_sample_count"] == 0


def test_single_comparable_sample_returns_unknown_stability() -> None:
    ranking = _ranking(scores=(0.9,))
    stability = _evaluate(ranking, _explanations(ranking)).artifact["projects"][0][
        "explanation_stability"
    ]

    assert stability["stability_pair_count"] == 0
    assert stability["mean_explanation_similarity"] is None


def test_evidence_coverage_counts_observed_categories_and_deduplicates_features() -> (
    None
):
    ranking = _ranking()
    result = _evaluate(ranking, _explanations(ranking))
    row = result.action_quality.iloc[0]

    for flag in (
        "has_identifier",
        "has_timestamp",
        "has_hotspot_evidence",
        "has_explanation_evidence",
        "has_signal_evidence",
    ):
        assert bool(row[flag])
    assert row["evidence_coverage_score"] == 1.0
    assert row["explanation_item_count"] == 2
    assert row["signal_item_count"] == 1
    assert row["evidence_item_count"] == 6
    coverage = result.artifact["projects"][0]["evidence_coverage"]
    assert coverage["mean_evidence_coverage_score"] == 1.0
    assert coverage["action_count"] == 2


@pytest.mark.parametrize("recurring_count", [0, 1])
def test_single_observation_or_zero_default_is_not_hotspot_evidence(
    recurring_count: int,
) -> None:
    ranking = _ranking(scores=(0.9,))
    actions = _actions(ranking)
    actions.loc[0, "dominant_signal"] = None
    actions.loc[0, "supporting_evidence"] = json.dumps(
        {"recurring_high_risk_count": recurring_count, "risk_signal_count": 0}
    )
    row = _evaluate(
        ranking, pd.DataFrame(columns=EXPLANATION_COLUMNS), actions
    ).action_quality.iloc[0]

    assert not bool(row["has_hotspot_evidence"])
    assert not bool(row["has_signal_evidence"])
    assert not bool(row["has_explanation_evidence"])
    assert row["evidence_coverage_score"] == pytest.approx(0.4)
    assert row["evidence_item_count"] == 2


def test_missing_identifiers_dates_and_evidence_are_not_invented() -> None:
    ranking = _ranking(scores=(0.9,))
    ranking.loc[0, ["file_path", "snapshot_date"]] = None
    actions = _actions(ranking)
    actions.loc[
        0,
        [
            "dominant_signal",
            "short_reason",
            "recommended_action",
            "supporting_evidence",
        ],
    ] = None
    result = _evaluate(ranking, pd.DataFrame(columns=EXPLANATION_COLUMNS), actions)
    row = result.action_quality.iloc[0]

    assert pd.isna(row["identifier"])
    assert pd.isna(row["snapshot_date"])
    assert row["evidence_coverage_score"] == 0.0
    assert row["evidence_item_count"] == 0
    assert row["specificity_score"] == 0.0
    assert row["specificity_label"] == "LOW"
    alignment = result.artifact["projects"][0]["risk_explanation_alignment"]
    assert alignment["explained_high_risk_count"] == 0
    assert alignment["mean_top_feature_share"] is None
    assert alignment["mean_explanation_strength"] is None
    json.dumps(result.artifact, allow_nan=False)


@pytest.mark.parametrize(
    ("identifier", "signal", "recommendation", "score", "label"),
    [
        (None, None, "Review code", 0.0, "LOW"),
        ("src/A.java", None, "Review code", 0.25, "LOW"),
        ("src/A.java", "code_churn", "Review code", 0.5, "MODERATE"),
        (
            "src/A.java",
            None,
            "Inspect recent edits for regression-test gaps.",
            0.75,
            "HIGH",
        ),
        (
            "src/A.java",
            "code_churn",
            "Inspect recent edits for regression-test gaps.",
            1.0,
            "HIGH",
        ),
    ],
)
def test_specificity_requires_concrete_targets_and_references(
    identifier: str | None,
    signal: str | None,
    recommendation: str,
    score: float,
    label: str,
) -> None:
    ranking = _ranking(scores=(0.9,))
    actions = _actions(ranking)
    actions.loc[0, ["identifier", "file_path"]] = identifier
    actions.loc[0, "dominant_signal"] = signal
    actions.loc[0, "short_reason"] = None
    actions.loc[0, "recommended_action"] = recommendation
    actions.loc[0, "supporting_evidence"] = "{}"

    row = _evaluate(
        ranking, pd.DataFrame(columns=EXPLANATION_COLUMNS), actions
    ).action_quality.iloc[0]

    assert row["specificity_score"] == score
    assert row["specificity_label"] == label


def test_empty_action_set_preserves_schema_and_unknown_action_means() -> None:
    ranking = _ranking()
    result = _evaluate(
        ranking, _explanations(ranking), pd.DataFrame(columns=DEVELOPER_ACTION_COLUMNS)
    )
    project = result.artifact["projects"][0]

    assert result.action_quality.empty
    assert tuple(result.action_quality.columns) == ACTION_QUALITY_COLUMNS
    assert project["evidence_coverage"]["action_count"] == 0
    assert project["evidence_coverage"]["mean_evidence_coverage_score"] is None
    assert project["action_specificity"]["mean_specificity_score"] is None
    assert project["risk_explanation_alignment"]["high_risk_sample_count"] == 2


def test_no_high_risk_samples_returns_unknown_alignment_rates() -> None:
    ranking = _ranking(scores=(0.1, 0.2))
    result = _evaluate(
        ranking, _explanations(ranking), pd.DataFrame(columns=DEVELOPER_ACTION_COLUMNS)
    )
    alignment = result.artifact["projects"][0]["risk_explanation_alignment"]

    assert alignment["high_risk_sample_count"] == 0
    assert alignment["explained_high_risk_count"] == 0
    assert alignment["explanation_coverage_rate"] is None
    assert alignment["mean_top_feature_share"] is None
    assert alignment["mean_explanation_strength"] is None


def test_strength_is_unknown_when_explanation_units_or_methods_differ() -> None:
    ranking = _ranking()
    explanations = _explanations(ranking)
    explanations.loc[explanations.index >= 2, "contribution_space"] = (
        "probability_delta"
    )
    explanations.loc[explanations.index >= 2, "explanation_method"] = (
        "tree_path_probability"
    )
    alignment = _evaluate(ranking, explanations).artifact["projects"][0][
        "risk_explanation_alignment"
    ]

    assert alignment["explained_high_risk_count"] == 2
    assert alignment["mean_top_feature_share"] == pytest.approx(0.75)
    assert alignment["mean_explanation_strength"] is None


def test_nonfinite_weights_are_unknown_and_json_contains_only_finite_numbers() -> None:
    ranking = _ranking(scores=(0.9,))
    explanations = _explanations(ranking)
    explanations.loc[:, "contribution"] = [np.nan, np.inf]
    explanations.loc[:, "absolute_contribution"] = [np.nan, np.inf]
    result = _evaluate(
        ranking, explanations, pd.DataFrame(columns=DEVELOPER_ACTION_COLUMNS)
    )
    alignment = result.artifact["projects"][0]["risk_explanation_alignment"]

    assert alignment["explained_high_risk_count"] == 0
    assert alignment["mean_explanation_strength"] is None
    json.dumps(result.artifact, allow_nan=False)


def test_cross_project_metrics_and_action_evidence_are_independent() -> None:
    alpha = _ranking()
    beta = _ranking("owner/beta", scores=(0.9, 0.4))
    ranking = pd.concat([alpha, beta], ignore_index=True)
    explanations = pd.concat(
        [
            _explanations(alpha),
            _explanations(beta, ((1.0, 0.0), (-1.0, 0.0))),
        ],
        ignore_index=True,
    )
    actions = _actions(ranking)
    actions.loc[actions["project"] == "owner/beta", "supporting_evidence"] = "{}"
    result = _evaluate(ranking, explanations, actions, experiment_type="cross_project")
    projects = {project["project"]: project for project in result.artifact["projects"]}

    assert set(projects) == {"owner/alpha", "owner/beta"}
    assert (
        projects["owner/alpha"]["risk_explanation_alignment"]["high_risk_sample_count"]
        == 2
    )
    assert (
        projects["owner/beta"]["risk_explanation_alignment"]["high_risk_sample_count"]
        == 1
    )
    assert projects["owner/alpha"]["explanation_stability"][
        "mean_explanation_similarity"
    ] == pytest.approx(1.0)
    assert projects["owner/beta"]["explanation_stability"][
        "mean_explanation_similarity"
    ] == pytest.approx(-1.0)
    beta_quality = result.action_quality.loc[
        result.action_quality["project"] == "owner/beta"
    ]
    assert not beta_quality["has_hotspot_evidence"].any()


def test_explanations_from_another_project_cannot_fill_missing_evidence() -> None:
    ranking = _ranking()
    foreign = _explanations(_ranking("owner/beta"))
    actions = _actions(ranking)
    actions.loc[:, "supporting_evidence"] = "{}"
    actions.loc[:, "dominant_signal"] = None
    result = _evaluate(ranking, foreign, actions, experiment_type="cross_project")
    project = next(
        row for row in result.artifact["projects"] if row["project"] == "owner/alpha"
    )

    assert project["risk_explanation_alignment"]["explained_high_risk_count"] == 0
    assert not result.action_quality["has_explanation_evidence"].any()


def test_json_and_csv_have_stable_public_schema() -> None:
    ranking = _ranking()
    result = _evaluate(ranking, _explanations(ranking))
    artifact = json.loads(json.dumps(result.artifact, allow_nan=False))
    csv = pd.read_csv(StringIO(result.action_quality.to_csv(index=False)))

    assert artifact["schema_version"] == ACTION_EVALUATION_SCHEMA_VERSION == 1
    assert {
        "project",
        "evaluation_counts",
        "explanation_stability",
        "evidence_coverage",
        "action_specificity",
        "risk_explanation_alignment",
        "limitations",
    } <= set(artifact["projects"][0])
    assert artifact["projects"][0]["limitations"]
    assert tuple(csv.columns) == ACTION_QUALITY_COLUMNS
    assert {
        "project",
        "rank",
        "tier",
        "identifier",
        "risk_score",
        "specificity_label",
    } <= set(csv)
    assert len(csv) == len(_actions(ranking))
    assert csv["evidence_coverage_score"].between(0, 1).all()
    assert csv["specificity_score"].between(0, 1).all()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    ranking = _ranking()
    with pytest.raises(ValueError, match="risk_threshold"):
        build_action_evaluation(
            ranking,
            _explanations(ranking),
            _actions(ranking),
            risk_threshold=threshold,
            experiment_type="within_project",
        )


def test_empty_inputs_preserve_unknown_project_and_header_only_csv() -> None:
    result = _evaluate(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert result.artifact["projects"][0]["project"] is None
    assert (
        result.artifact["projects"][0]["evaluation_counts"]["evaluated_sample_count"]
        == 0
    )
    assert len(pd.read_csv(StringIO(result.action_quality.to_csv(index=False)))) == 0


@pytest.mark.parametrize("column", ["model", "evaluation_mode"])
def test_explanation_join_respects_evaluated_model_and_mode(column: str) -> None:
    ranking = _ranking()
    explanations = _explanations(ranking)
    explanations[column] = "foreign"
    actions = _actions(ranking)
    actions["supporting_evidence"] = "{}"
    result = _evaluate(ranking, explanations, actions)
    assert (
        result.artifact["projects"][0]["risk_explanation_alignment"][
            "explained_high_risk_count"
        ]
        == 0
    )
    assert not result.action_quality["has_explanation_evidence"].any()


def test_missing_middle_explanation_does_not_create_a_neighboring_pair() -> None:
    ranking = _ranking()
    ranking.loc[2] = ranking.iloc[1]
    ranking.loc[2, "snapshot_date"] = "2024-03-31"
    explanations = _explanations(ranking.iloc[[0, 2]])
    stability = _evaluate(ranking, explanations).artifact["projects"][0][
        "explanation_stability"
    ]
    assert stability["stability_pair_count"] == 0


def test_duplicate_sample_identities_do_not_manufacture_explanation_matches() -> None:
    ranking = _ranking(scores=(0.9,))
    explanations = _explanations(ranking)
    duplicated = pd.concat([ranking, ranking], ignore_index=True)
    actions = _actions(duplicated)
    actions["supporting_evidence"] = "{}"
    result = _evaluate(duplicated, explanations, actions)
    assert not result.action_quality["has_explanation_evidence"].any()
    assert (
        result.artifact["projects"][0]["risk_explanation_alignment"][
            "explained_high_risk_count"
        ]
        == 0
    )


def test_generic_action_does_not_reference_unmentioned_external_features() -> None:
    ranking = _ranking()
    actions = _actions(ranking)
    actions["dominant_signal"] = None
    actions["short_reason"] = None
    actions["supporting_evidence"] = "{}"
    actions["recommended_action"] = "Review the detailed prediction output manually."
    result = _evaluate(ranking, _explanations(ranking), actions)
    assert result.action_quality["has_explanation_evidence"].all()
    assert result.action_quality["specificity_score"].eq(0.25).all()
    assert result.action_quality["specificity_label"].eq("LOW").all()


def test_embedded_evidence_survives_truncation_without_reconstructing_vectors() -> None:
    ranking = _ranking()
    explanations = _explanations(ranking)
    explanations = explanations.loc[explanations["feature"] == "commit_count"]
    result = _evaluate(ranking, explanations)
    assert result.action_quality["explanation_item_count"].eq(2).all()
    items = json.loads(result.action_quality.iloc[0]["supporting_explanations"])
    assert {item["source"] for item in items} == {
        "prediction_explanations",
        "developer_actions.supporting_evidence",
    }
    alignment = result.artifact["projects"][0]["risk_explanation_alignment"]
    assert alignment["mean_top_feature_share"] == 1.0
    assert alignment["mean_explanation_strength"] == pytest.approx(0.15)


def test_nonfinite_attached_evidence_is_serialized_as_null() -> None:
    ranking = _ranking(scores=(0.9,))
    actions = _actions(ranking)
    actions.loc[0, "supporting_evidence"] = (
        '{"contribution": Infinity, "risk_signal_count": NaN}'
    )
    row = _evaluate(ranking, pd.DataFrame(), actions).action_quality.iloc[0]
    assert not row["has_explanation_evidence"]
    assert json.loads(row["supporting_evidence"]) == {
        "contribution": None,
        "risk_signal_count": None,
    }


def test_threshold_is_inclusive_and_zero_vectors_are_not_explained_high_risk() -> None:
    ranking = _ranking(scores=(0.5, 0.49))
    result = _evaluate(ranking, _explanations(ranking, ((0.0, 0.0), (0.3, -0.1))))
    alignment = result.artifact["projects"][0]["risk_explanation_alignment"]
    assert alignment["high_risk_sample_count"] == 1
    assert alignment["explained_high_risk_count"] == 0
    assert alignment["explanation_coverage_rate"] == 0.0
    assert alignment["mean_top_feature_share"] is None


def test_multiple_models_at_same_timestamp_do_not_make_order_dependent_pairs() -> None:
    ranking = _ranking()
    other = ranking.iloc[:1].copy()
    other["model"] = "other-model"
    explanations = pd.concat([_explanations(ranking), _explanations(other)])
    ranking = pd.concat([ranking, other], ignore_index=True)
    first = _evaluate(ranking, explanations)
    second = _evaluate(ranking.iloc[::-1], explanations.iloc[::-1])
    a = first.artifact["projects"][0]["explanation_stability"]
    b = second.artifact["projects"][0]["explanation_stability"]
    assert a == b
    assert a["stability_pair_count"] == 0
