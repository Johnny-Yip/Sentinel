from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sentinel.evaluation.explain import (
    EXPLANATION_COLUMNS,
    build_explanation_summary,
    explain_predictions,
)
from sentinel.ml.evaluation import positive_class_scores


class AdditiveProbabilityClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        values = np.asarray(features, dtype=float)
        margin = values[:, 0] + values[:, 1]
        positive = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1.0 - positive, positive])


class ConstantProbabilityClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        positive = np.full(len(features), 0.4)
        return np.column_stack([1.0 - positive, positive])


def _with_metadata(features: pd.DataFrame) -> pd.DataFrame:
    samples = features.copy()
    samples.insert(0, "snapshot_date", "2024-01-31")
    samples.insert(
        0,
        "file_path",
        [f"src/File{index}.java" for index in range(len(samples))],
    )
    samples.insert(0, "repository", "owner/example")
    samples["defect_next_90_days"] = np.arange(len(samples)) % 2
    return samples


def test_linear_contributions_use_transformed_values_and_coefficients() -> None:
    features = pd.DataFrame(
        {
            "activity": np.arange(8, dtype=float),
            "recency": [8.0, 7.0, 7.0, 6.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pipeline = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", StandardScaler(), ["activity", "recency"])],
                    verbose_feature_names_out=False,
                ),
            ),
            ("model", LogisticRegression(solver="liblinear", random_state=4)),
        ]
    ).fit(features, target)
    scores, _ = positive_class_scores(pipeline, features)

    explanations = explain_predictions(
        pipeline,
        _with_metadata(features),
        ["activity", "recency"],
        scores,
        model="logistic_regression",
        evaluation_mode="test",
        decision_threshold=0.5,
    )

    transformed = pipeline.named_steps["preprocess"].transform(features)
    coefficients = pipeline.named_steps["model"].coef_[0]
    expected = transformed * coefficients
    for sample_index, sample_id in enumerate(explanations["sample_id"].unique()):
        rows = explanations.loc[explanations["sample_id"] == sample_id].set_index(
            "feature"
        )
        for feature_index, feature in enumerate(("activity", "recency")):
            assert rows.loc[feature, "contribution"] == pytest.approx(
                expected[sample_index, feature_index]
            )
            assert rows.loc[feature, "model_input_value"] == pytest.approx(
                transformed[sample_index, feature_index]
            )
            assert rows.loc[feature, "feature_value"] == features.loc[
                sample_index, feature
            ]
    assert set(explanations["direction"]) >= {
        "increases_risk",
        "decreases_risk",
    }
    assert explanations["explanation_method"].eq(
        "linear_coefficient_transformed"
    ).all()


def test_tree_path_explanations_are_deterministic_and_reconstruct_probability() -> None:
    features = pd.DataFrame(
        {"activity": [0.0, 1.0, 2.0, 3.0], "recency": [3.0, 2.0, 1.0, 0.0]}
    )
    target = np.array([0, 0, 1, 1])
    forest = RandomForestClassifier(
        n_estimators=7, max_depth=2, random_state=9
    ).fit(features, target)
    scores, _ = positive_class_scores(forest, features)
    samples = _with_metadata(features)

    first = explain_predictions(
        forest,
        samples,
        list(features.columns),
        scores,
        model="decision_tree",
        evaluation_mode="test",
        decision_threshold=0.5,
    )
    second = explain_predictions(
        forest,
        samples,
        list(features.columns),
        scores,
        model="decision_tree",
        evaluation_mode="test",
        decision_threshold=0.5,
    )

    pd.testing.assert_frame_equal(first, second)
    root_probability = np.mean(
        [
            estimator.tree_.value[0].reshape(-1)[1]
            / estimator.tree_.value[0].sum()
            for estimator in forest.estimators_
        ]
    )
    contribution_sums = first.groupby("sample_id", sort=False)[
        "contribution"
    ].sum()
    np.testing.assert_allclose(contribution_sums, scores - root_probability)
    assert first["explanation_method"].eq("tree_path_probability").all()


def test_unsupported_model_fallback_and_tie_breaking_are_deterministic() -> None:
    features = pd.DataFrame(
        {"zeta": [1.0, 0.0, -1.0], "alpha": [1.0, 0.0, -1.0]}
    )
    samples = _with_metadata(features)
    classifier = AdditiveProbabilityClassifier()
    scores, _ = positive_class_scores(classifier, features)

    first = explain_predictions(
        classifier,
        samples,
        ["zeta", "alpha"],
        scores,
        model="unsupported",
        evaluation_mode="test",
        decision_threshold=0.5,
        top_k=1,
    )
    second = explain_predictions(
        classifier,
        samples,
        ["zeta", "alpha"],
        scores,
        model="unsupported",
        evaluation_mode="test",
        decision_threshold=0.5,
        top_k=1,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["explanation_method"].eq("local_median_perturbation").all()
    assert first.iloc[0]["feature"] == "alpha"
    assert first.iloc[0]["normalized_contribution"] == pytest.approx(1.0)


def test_normalized_magnitudes_sum_to_one_or_zero_per_sample() -> None:
    features = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [2.0, 1.0, 0.0]})
    samples = _with_metadata(features)
    classifier = AdditiveProbabilityClassifier()
    scores, _ = positive_class_scores(classifier, features)
    explanations = explain_predictions(
        classifier,
        samples,
        ["a", "b"],
        scores,
        model="additive",
        evaluation_mode="test",
        decision_threshold=0.5,
    )

    grouped = explanations.groupby("sample_id", sort=False)
    for _, rows in grouped:
        normalized_sum = rows["normalized_contribution"].sum()
        if rows["absolute_contribution"].sum() > 0.0:
            assert normalized_sum == pytest.approx(1.0)
        else:
            assert normalized_sum == 0.0

    constant = ConstantProbabilityClassifier()
    constant_scores, _ = positive_class_scores(constant, features)
    neutral = explain_predictions(
        constant,
        samples,
        ["a", "b"],
        constant_scores,
        model="constant",
        evaluation_mode="test",
        decision_threshold=0.5,
    )
    assert neutral["contribution"].eq(0.0).all()
    assert neutral["normalized_contribution"].eq(0.0).all()
    assert neutral["direction"].eq("neutral").all()


def test_explanation_summary_contains_counts_magnitudes_and_metadata() -> None:
    features = pd.DataFrame(
        {"activity": [0.0, 1.0, 2.0], "recency": [2.0, 1.0, 0.0]}
    )
    samples = _with_metadata(features)
    classifier = AdditiveProbabilityClassifier()
    scores, _ = positive_class_scores(classifier, features)
    explanations = explain_predictions(
        classifier,
        samples,
        ["activity", "recency"],
        scores,
        model="fallback",
        evaluation_mode="test",
        decision_threshold=0.5,
    )

    summary = build_explanation_summary(explanations)

    assert list(explanations.columns) == list(EXPLANATION_COLUMNS)
    assert summary["total_explained_samples"] == 3
    assert summary["total_explained_feature_contributions"] == 6
    assert summary["most_common_risk_increasing_features"]
    assert summary["most_common_risk_decreasing_features"]
    assert len(summary["average_absolute_contribution_by_feature"]) == 2
    assert summary["explanation_metadata"]["methods"] == [
        "local_median_perturbation"
    ]
