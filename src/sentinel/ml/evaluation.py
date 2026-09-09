"""Imbalance-aware metrics and validation-only threshold selection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
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
