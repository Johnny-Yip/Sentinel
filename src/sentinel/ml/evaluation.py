"""Reusable classification metrics, explainability, and threshold selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_THRESHOLD = 0.5


def calculate_metrics(
    y_true: np.ndarray,
    positive_scores: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Calculate rare-positive classification metrics without warning on edge cases."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(positive_scores, dtype=float)
    if truth.shape != scores.shape:
        raise ValueError("Target values and probability scores must have equal shapes.")
    if truth.size == 0:
        raise ValueError("Metrics cannot be calculated for an empty dataset.")
    if not np.isfinite(scores).all():
        raise ValueError("Probability scores must be finite.")
    if threshold < 0 or threshold > 1:
        raise ValueError("Classification threshold must be between 0 and 1.")

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    class_count = np.unique(truth).size
    positive_count = int(truth.sum())

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(truth, scores)) if class_count == 2 else None
        ),
        "pr_auc": (
            float(average_precision_score(truth, scores))
            if positive_count > 0
            else None
        ),
        "confusion_matrix": matrix.astype(int).tolist(),
        "positive_prediction_rate": float(predictions.mean()),
    }


def extract_feature_importance(
    pipeline: Any,
    feature_columns: Sequence[str],
    evaluation_features: Any,
    evaluation_target: np.ndarray,
    *,
    random_state: int = 42,
    n_repeats: int = 5,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    """Return ranked native or permutation importance records for a classifier.

    Linear coefficients and tree-style ``feature_importances_`` are preferred.
    Other estimators fall back to deterministic permutation importance on the
    supplied evaluation partition. The absolute ``importance`` value is used
    only for ranking; ``effect`` retains the signed coefficient or score drop.
    """
    if n_repeats <= 0:
        raise ValueError("Permutation importance repeats must be positive.")
    if limit is not None and limit <= 0:
        raise ValueError("Feature importance limit must be positive.")

    estimator = getattr(pipeline, "named_steps", {}).get("model", pipeline)
    preprocessor = getattr(pipeline, "named_steps", {}).get("preprocess")
    transformed_columns = list(feature_columns)
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        transformed_columns = list(preprocessor.get_feature_names_out())

    standard_deviations: np.ndarray | None = None
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients.reshape(1, -1)
        if coefficients.shape[0] == 1:
            effects: list[float | None] = coefficients[0].tolist()
            magnitudes = np.abs(coefficients[0])
        else:
            effects = [None] * coefficients.shape[1]
            magnitudes = np.mean(np.abs(coefficients), axis=0)
        names = transformed_columns
        method = "native_coefficient"
    elif hasattr(estimator, "feature_importances_"):
        native = np.asarray(estimator.feature_importances_, dtype=float)
        effects = native.tolist()
        magnitudes = np.abs(native)
        names = transformed_columns
        method = "native_feature_importance"
    else:
        truth = np.asarray(evaluation_target, dtype=int)
        if truth.size == 0:
            raise ValueError("Feature importance cannot use an empty dataset.")
        scoring = (
            "average_precision"
            if np.unique(truth).size == 2 and int(truth.sum()) > 0
            else "accuracy"
        )
        permutation = permutation_importance(
            pipeline,
            evaluation_features,
            truth,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=1,
        )
        raw_effects = np.asarray(permutation.importances_mean, dtype=float)
        effects = raw_effects.tolist()
        magnitudes = np.abs(raw_effects)
        standard_deviations = np.asarray(permutation.importances_std, dtype=float)
        names = list(feature_columns)
        method = f"permutation_{scoring}"

    if len(names) != len(magnitudes):
        raise ValueError(
            "Feature names and extracted importance values must have equal lengths."
        )

    records: list[dict[str, Any]] = []
    for index, (feature, magnitude, effect) in enumerate(
        zip(names, magnitudes, effects, strict=True)
    ):
        direction = "not_applicable"
        if method == "native_coefficient" and effect is not None:
            direction = (
                "positive"
                if effect > 0
                else "negative"
                if effect < 0
                else "zero"
            )
        records.append(
            {
                "feature": str(feature),
                "importance": float(magnitude),
                "effect": float(effect) if effect is not None else None,
                "direction": direction,
                "method": method,
                "standard_deviation": (
                    float(standard_deviations[index])
                    if standard_deviations is not None
                    else None
                ),
            }
        )

    records.sort(key=lambda record: (-record["importance"], record["feature"]))
    if limit is not None:
        records = records[:limit]
    for rank, record in enumerate(records, start=1):
        record["rank"] = rank
    return records


def select_f1_threshold(
    y_validation: np.ndarray,
    validation_scores: np.ndarray,
    default_threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """Choose the validation threshold maximizing positive-class F1."""
    truth = np.asarray(y_validation, dtype=int)
    scores = np.asarray(validation_scores, dtype=float)
    if truth.shape != scores.shape or truth.size == 0:
        raise ValueError("Validation targets and scores must be non-empty and aligned.")
    if not np.isfinite(scores).all():
        raise ValueError("Validation probability scores must be finite.")
    if int(truth.sum()) == 0:
        return float(default_threshold)

    precision, recall, thresholds = precision_recall_curve(truth, scores)
    if thresholds.size == 0:
        return float(default_threshold)

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    best_index = int(np.flatnonzero(f1_values == f1_values.max())[0])
    # Parallel estimators can differ at the final binary floating-point bit even
    # with a fixed seed. Normalizing downward beyond meaningful probability
    # precision makes persisted thresholds exactly reproducible without moving a
    # selected score just above itself (important for constant-probability dummy
    # models and the inclusive ``scores >= threshold`` decision rule).
    clipped = float(np.clip(thresholds[best_index], 0.0, 1.0))
    precision = 10**14
    return float(np.floor(clipped * precision) / precision)
