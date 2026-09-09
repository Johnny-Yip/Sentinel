"""Deterministic local explanations for evaluated Sentinel samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sentinel.ml.data import (
    DATE_COLUMN,
    PATH_COLUMN,
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
)
from sentinel.ml.evaluation import positive_class_scores


EXPLANATION_EPSILON = 1e-12
EXPLANATION_COLUMNS = (
    "sample_id",
    "project",
    "file_path",
    "snapshot_date",
    "risk_score",
    "predicted_label",
    "true_label",
    "model",
    "evaluation_mode",
    "explanation_method",
    "contribution_space",
    "feature",
    "feature_value",
    "model_input_value",
    "contribution",
    "absolute_contribution",
    "normalized_contribution",
    "direction",
    "feature_rank",
)
_RANKING_KEY_COLUMNS = (
    "project",
    "file_path",
    "snapshot_date",
    "model",
    "evaluation_mode",
)
_EXPLANATION_SAMPLE_KEY_COLUMNS = ("sample_id", "model", "evaluation_mode")


def _pipeline_parts(classifier: Any) -> tuple[Any | None, Any]:
    steps = getattr(classifier, "named_steps", {})
    return steps.get("preprocess"), steps.get("model", classifier)


def _dense_2d(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        values = values.toarray()
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Transformed explanation features must be a finite matrix.")
    return array


def _transformed_features(
    classifier: Any,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[Any, np.ndarray, list[str], np.ndarray]:
    preprocessor, estimator = _pipeline_parts(classifier)
    if preprocessor is None:
        transformed = features.to_numpy(dtype=float)
        names = list(feature_columns)
    else:
        transformed = preprocessor.transform(features)
        get_names = getattr(preprocessor, "get_feature_names_out", None)
        names = (
            [str(name) for name in get_names()]
            if callable(get_names)
            else list(feature_columns)
        )
    matrix = _dense_2d(transformed)
    if matrix.shape[1] != len(names):
        raise ValueError(
            "Transformed feature names must align with the model input columns."
        )

    raw = features.to_numpy(dtype=float)
    display_values = raw if names == list(feature_columns) else matrix
    return estimator, matrix, names, display_values


def _positive_index(estimator: Any) -> tuple[np.ndarray, int]:
    classes = np.asarray(getattr(estimator, "classes_", []))
    indices = np.flatnonzero(classes == 1)
    if indices.size != 1:
        raise ValueError("The fitted estimator must expose positive class 1.")
    return classes, int(indices[0])


def _linear_contributions(
    estimator: Any, transformed: np.ndarray
) -> np.ndarray:
    classes, positive_index = _positive_index(estimator)
    coefficients = np.asarray(estimator.coef_, dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)

    if coefficients.shape[0] == 1 and len(classes) == 2:
        # sklearn's one-row binary coefficient is oriented toward classes_[1].
        sign = 1.0 if positive_index == 1 else -1.0
        positive_coefficients = sign * coefficients[0]
    elif coefficients.shape[0] == len(classes):
        positive_coefficients = coefficients[positive_index]
    else:
        raise ValueError("Linear coefficients must align with estimator classes.")
    if transformed.shape[1] != len(positive_coefficients):
        raise ValueError(
            "Transformed feature names and linear coefficients must align."
        )
    return transformed * positive_coefficients


def _classifier_trees(estimator: Any) -> list[Any] | None:
    if hasattr(estimator, "tree_") and hasattr(estimator, "classes_"):
        return [estimator]
    members = getattr(estimator, "estimators_", None)
    if members is None:
        return None
    trees = list(np.asarray(members, dtype=object).ravel())
    if not trees or not all(
        hasattr(tree, "tree_") and hasattr(tree, "classes_") for tree in trees
    ):
        return None
    return trees


def _tree_contributions(
    trees: Sequence[Any], transformed: np.ndarray
) -> np.ndarray:
    contributions = np.zeros_like(transformed, dtype=float)
    for tree_estimator in trees:
        tree = tree_estimator.tree_
        classes, positive_index = _positive_index(tree_estimator)
        values = np.asarray(tree.value, dtype=float)
        if (
            values.ndim != 3
            or values.shape[1] != 1
            or values.shape[2] != len(classes)
        ):
            raise ValueError("Tree node values must align with estimator classes.")
        weights = values[:, 0, :]
        totals = weights.sum(axis=1)
        probabilities = np.divide(
            weights[:, positive_index],
            totals,
            out=np.zeros(tree.node_count, dtype=float),
            where=totals > 0.0,
        )

        parents = np.full(tree.node_count, -1, dtype=int)
        internal = np.flatnonzero(tree.children_left != tree.children_right)
        parents[tree.children_left[internal]] = internal
        parents[tree.children_right[internal]] = internal
        non_root = np.flatnonzero(parents >= 0)
        parent_features = tree.feature[parents[non_root]].astype(int)
        if (
            (parent_features < 0).any()
            or (parent_features >= transformed.shape[1]).any()
        ):
            raise ValueError("Tree split feature does not align with model input.")
        node_contributions = np.zeros(
            (tree.node_count, transformed.shape[1]), dtype=float
        )
        node_contributions[non_root, parent_features] = (
            probabilities[non_root] - probabilities[parents[non_root]]
        )
        path = tree_estimator.decision_path(transformed)
        contributions += np.asarray(path @ node_contributions)
    return contributions / len(trees)


def _fallback_contributions(
    classifier: Any,
    features: pd.DataFrame,
    reference_features: pd.DataFrame,
    feature_columns: Sequence[str],
    risk_scores: np.ndarray,
) -> np.ndarray:
    contributions = np.zeros((len(features), len(feature_columns)), dtype=float)
    reference_values = reference_features.loc[:, feature_columns].median(axis=0)
    if reference_values.isna().any():
        raise ValueError("Perturbation reference feature medians must be finite.")

    for feature_index, feature in enumerate(feature_columns):
        perturbed = features.copy()
        perturbed.loc[:, feature] = float(reference_values[feature])
        perturbed_scores, _ = positive_class_scores(classifier, perturbed)
        contributions[:, feature_index] = risk_scores - perturbed_scores
    return contributions


def _expanded_threshold(threshold: float, length: int) -> np.ndarray:
    value = float(threshold)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    return np.full(length, value, dtype=float)


def _sample_metadata(samples: pd.DataFrame, column: str, default: Any) -> list[Any]:
    if column not in samples:
        return [default] * len(samples)
    if column == DATE_COLUMN:
        return pd.to_datetime(samples[column]).dt.strftime("%Y-%m-%d").tolist()
    return samples[column].tolist()


def explain_predictions(
    classifier: Any,
    samples: pd.DataFrame,
    feature_columns: Sequence[str],
    risk_scores: Sequence[float] | np.ndarray,
    *,
    model: str,
    evaluation_mode: str,
    decision_threshold: float,
    reference_features: pd.DataFrame | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Explain each sample with model-aligned deterministic local contributions.

    Linear contributions are in log-odds/margin space. Classifier-tree path
    contributions and unsupported-estimator perturbations are probability
    deltas. ``feature_value`` stays in the raw feature space when preprocessing
    is one-to-one; ``model_input_value`` always records the transformed value.
    """
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("Prediction explanations require at least one feature.")
    if top_k is not None and top_k <= 0:
        raise ValueError("explain_top_k must be positive.")
    missing = [column for column in feature_columns if column not in samples]
    if missing:
        raise ValueError(
            "Explanation samples are missing features: " + ", ".join(missing)
        )
    if samples.empty:
        raise ValueError("Prediction explanations require at least one sample.")

    scores = np.asarray(risk_scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(samples):
        raise ValueError("Risk scores must contain one value per explained sample.")
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Explanation risk scores must be finite and between 0 and 1.")
    thresholds = _expanded_threshold(decision_threshold, len(samples))

    features = samples.loc[:, feature_columns].copy()
    estimator, transformed, transformed_names, display_values = _transformed_features(
        classifier, features, feature_columns
    )
    trees = _classifier_trees(estimator)
    if hasattr(estimator, "coef_"):
        contributions = _linear_contributions(estimator, transformed)
        names = transformed_names
        model_values = transformed
        method = "linear_coefficient_transformed"
        contribution_space = (
            "log_odds"
            if estimator.__class__.__name__ == "LogisticRegression"
            else "decision_margin"
        )
    elif trees is not None:
        contributions = _tree_contributions(trees, transformed)
        names = transformed_names
        model_values = transformed
        method = "tree_path_probability"
        contribution_space = "probability_delta"
    else:
        reference = samples if reference_features is None else reference_features
        missing_reference = [
            column for column in feature_columns if column not in reference
        ]
        if missing_reference:
            raise ValueError(
                "Explanation reference data are missing features: "
                + ", ".join(missing_reference)
            )
        contributions = _fallback_contributions(
            classifier,
            features,
            reference.loc[:, feature_columns],
            feature_columns,
            scores,
        )
        names = feature_columns
        display_values = features.to_numpy(dtype=float)
        model_values = (
            transformed
            if transformed_names == feature_columns
            and transformed.shape == display_values.shape
            else display_values
        )
        method = "local_median_perturbation"
        contribution_space = "probability_delta"

    contributions = np.asarray(contributions, dtype=float)
    if contributions.shape != (len(samples), len(names)):
        raise ValueError("Local contributions must align with samples and features.")
    if not np.isfinite(contributions).all():
        raise ValueError("Local contributions must be finite.")
    contributions[np.abs(contributions) <= EXPLANATION_EPSILON] = 0.0

    projects = _sample_metadata(samples, REPOSITORY_COLUMN, "")
    paths = _sample_metadata(samples, PATH_COLUMN, "")
    dates = _sample_metadata(samples, DATE_COLUMN, "")
    truths = _sample_metadata(samples, TARGET_COLUMN, pd.NA)
    sample_ids = [
        (
            f"{projects[index]}|{paths[index]}|{dates[index]}"
            if projects[index] or paths[index] or dates[index]
            else str(index + 1)
        )
        for index in range(len(samples))
    ]
    rows: list[dict[str, Any]] = []
    for sample_index in range(len(samples)):
        order = sorted(
            range(len(names)),
            key=lambda index: (-abs(contributions[sample_index, index]), names[index]),
        )
        selected = order if top_k is None else order[:top_k]
        denominator = float(
            np.abs(contributions[sample_index, selected]).sum()
        )
        for rank, feature_index in enumerate(selected, start=1):
            contribution = float(contributions[sample_index, feature_index])
            rows.append(
                {
                    "sample_id": sample_ids[sample_index],
                    "project": str(projects[sample_index]),
                    "file_path": str(paths[sample_index]),
                    "snapshot_date": str(dates[sample_index]),
                    "risk_score": float(scores[sample_index]),
                    "predicted_label": int(
                        scores[sample_index] >= thresholds[sample_index]
                    ),
                    "true_label": (
                        int(truths[sample_index])
                        if not pd.isna(truths[sample_index])
                        else pd.NA
                    ),
                    "model": str(model),
                    "evaluation_mode": str(evaluation_mode),
                    "explanation_method": method,
                    "contribution_space": contribution_space,
                    "feature": str(names[feature_index]),
                    "feature_value": float(
                        display_values[sample_index, feature_index]
                    ),
                    "model_input_value": float(
                        model_values[sample_index, feature_index]
                    ),
                    "contribution": contribution,
                    "absolute_contribution": abs(contribution),
                    "normalized_contribution": (
                        abs(contribution) / denominator
                        if denominator > 0.0
                        else 0.0
                    ),
                    "direction": (
                        "increases_risk"
                        if contribution > 0.0
                        else "decreases_risk"
                        if contribution < 0.0
                        else "neutral"
                    ),
                    "feature_rank": rank,
                }
            )
    return pd.DataFrame(rows, columns=EXPLANATION_COLUMNS)


def limit_explanations(
    explanations: pd.DataFrame, top_k: int | None
) -> pd.DataFrame:
    """Limit explanation rows per sample and renormalize retained magnitudes."""
    if top_k is None:
        return explanations.copy()
    if top_k <= 0:
        raise ValueError("explain_top_k must be positive.")
    limited = explanations.loc[explanations["feature_rank"] <= top_k].copy()
    denominators = limited.groupby(
        list(_EXPLANATION_SAMPLE_KEY_COLUMNS), sort=False
    )[
        "absolute_contribution"
    ].transform("sum")
    limited["normalized_contribution"] = np.divide(
        limited["absolute_contribution"],
        denominators,
        out=np.zeros(len(limited), dtype=float),
        where=denominators.to_numpy() > 0.0,
    )
    return limited.reset_index(drop=True).loc[:, EXPLANATION_COLUMNS]


def sort_explanations(explanations: pd.DataFrame) -> pd.DataFrame:
    """Order sample groups like the risk ranking and features by local rank."""
    return explanations.sort_values(
        [
            "risk_score",
            "project",
            "file_path",
            "snapshot_date",
            "model",
            "feature_rank",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True).loc[:, EXPLANATION_COLUMNS]


def add_top_explanations_to_ranking(
    ranking: pd.DataFrame, explanations: pd.DataFrame
) -> pd.DataFrame:
    """Add the strongest increasing and decreasing local drivers to a ranking."""
    grouped = {
        key: group
        for key, group in explanations.groupby(
            list(_RANKING_KEY_COLUMNS), sort=False
        )
    }
    result = ranking.copy()
    risk_features: list[Any] = []
    risk_contributions: list[float] = []
    protective_features: list[Any] = []
    protective_contributions: list[float] = []
    for row in result.itertuples(index=False):
        key = tuple(getattr(row, column) for column in _RANKING_KEY_COLUMNS)
        group = grouped.get(key)
        increasing = (
            group.loc[group["contribution"] > 0.0]
            if group is not None
            else pd.DataFrame()
        )
        decreasing = (
            group.loc[group["contribution"] < 0.0]
            if group is not None
            else pd.DataFrame()
        )
        if not increasing.empty:
            top = increasing.sort_values(
                ["contribution", "feature"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]
            risk_features.append(str(top["feature"]))
            risk_contributions.append(float(top["contribution"]))
        else:
            risk_features.append(pd.NA)
            risk_contributions.append(float("nan"))
        if not decreasing.empty:
            top = decreasing.sort_values(
                ["contribution", "feature"],
                ascending=[True, True],
                kind="stable",
            ).iloc[0]
            protective_features.append(str(top["feature"]))
            protective_contributions.append(float(top["contribution"]))
        else:
            protective_features.append(pd.NA)
            protective_contributions.append(float("nan"))

    result["top_risk_feature"] = risk_features
    result["top_risk_contribution"] = risk_contributions
    result["top_protective_feature"] = protective_features
    result["top_protective_contribution"] = protective_contributions
    return result


def _direction_counts(
    explanations: pd.DataFrame, direction: str
) -> list[dict[str, Any]]:
    counts = (
        explanations.loc[explanations["direction"] == direction, "feature"]
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="sample_count")
        .sort_values(
            ["sample_count", "feature"],
            ascending=[False, True],
            kind="stable",
        )
    )
    return [
        {"feature": str(row.feature), "sample_count": int(row.sample_count)}
        for row in counts.head(10).itertuples(index=False)
    ]


def build_explanation_summary(
    explanations: pd.DataFrame, *, explain_top_k: int | None = None
) -> dict[str, Any]:
    """Summarize local direction frequency, magnitude, and method metadata."""
    if explanations.empty:
        return {
            "total_explained_samples": 0,
            "total_explained_feature_contributions": 0,
            "most_common_risk_increasing_features": [],
            "most_common_risk_decreasing_features": [],
            "average_absolute_contribution_by_feature": [],
            "explanation_metadata": {
                "models": [],
                "methods": [],
                "contribution_spaces": [],
                "deterministic": True,
                "normalization": "absolute_magnitude_sum_per_sample",
                "feature_ordering": "absolute_contribution_desc_then_feature_name",
                "explain_top_k": explain_top_k,
            },
        }

    average = (
        explanations.groupby("feature", sort=False)["absolute_contribution"]
        .mean()
        .rename("average_absolute_contribution")
        .reset_index()
        .sort_values(
            ["average_absolute_contribution", "feature"],
            ascending=[False, True],
            kind="stable",
        )
    )
    method_rows: list[dict[str, Any]] = []
    for keys, group in explanations.groupby(
        ["model", "explanation_method", "contribution_space"], sort=True
    ):
        model, method, contribution_space = keys
        method_rows.append(
            {
                "model": str(model),
                "method": str(method),
                "contribution_space": str(contribution_space),
                "explained_samples": int(
                    group.drop_duplicates(
                        list(_EXPLANATION_SAMPLE_KEY_COLUMNS)
                    ).shape[0]
                ),
            }
        )
    return {
        "total_explained_samples": int(
            explanations.drop_duplicates(
                list(_EXPLANATION_SAMPLE_KEY_COLUMNS)
            ).shape[0]
        ),
        "total_explained_feature_contributions": len(explanations),
        "most_common_risk_increasing_features": _direction_counts(
            explanations, "increases_risk"
        ),
        "most_common_risk_decreasing_features": _direction_counts(
            explanations, "decreases_risk"
        ),
        "average_absolute_contribution_by_feature": [
            {
                "feature": str(row.feature),
                "average_absolute_contribution": float(
                    row.average_absolute_contribution
                ),
            }
            for row in average.itertuples(index=False)
        ],
        "explanation_metadata": {
            "models": sorted(explanations["model"].unique().tolist()),
            "methods": sorted(explanations["explanation_method"].unique().tolist()),
            "contribution_spaces": sorted(
                explanations["contribution_space"].unique().tolist()
            ),
            "model_methods": method_rows,
            "deterministic": True,
            "fallback_reference": (
                "per_evaluation_partition_feature_medians_when_used"
            ),
            "normalization": "absolute_magnitude_sum_per_sample",
            "feature_ordering": "absolute_contribution_desc_then_feature_name",
            "explain_top_k": explain_top_k,
        },
    }
