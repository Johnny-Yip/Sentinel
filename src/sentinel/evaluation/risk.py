"""Normalized sample-level risk scoring, ranking, and summaries."""

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


DEFAULT_TOP_RISK = 10
DEFAULT_RISK_THRESHOLD = 0.5
RISK_RANKING_COLUMNS = (
    "rank",
    "risk_score",
    "predicted_label",
    "true_label",
    "project",
    "file_path",
    "snapshot_date",
    "model",
    "evaluation_mode",
    "score_method",
    "decision_threshold",
    "top_risk_feature",
    "top_risk_contribution",
    "top_protective_feature",
    "top_protective_contribution",
)


def calculate_risk_scores(
    classifier: Any, features: Any
) -> tuple[np.ndarray, str]:
    """Score samples with probability estimates or normalized decision margins."""
    return positive_class_scores(classifier, features)


def _expanded(values: Any, length: int, name: str) -> list[Any]:
    if isinstance(values, str) or np.isscalar(values):
        return [values] * length
    expanded = list(values)
    if len(expanded) != length:
        raise ValueError(f"{name} must contain one value per evaluated sample.")
    return expanded


def build_risk_ranking(
    samples: pd.DataFrame,
    risk_scores: Sequence[float] | np.ndarray,
    *,
    model: str | Sequence[str],
    evaluation_mode: str,
    score_method: str | Sequence[str],
    decision_threshold: float | Sequence[float],
) -> pd.DataFrame:
    """Build a deterministic highest-to-lowest ranking for evaluated samples."""
    scores = np.asarray(risk_scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(samples):
        raise ValueError("Risk scores must contain one value per evaluated sample.")
    if not np.isfinite(scores).all():
        raise ValueError("Risk scores must be finite.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Risk scores must be between 0 and 1.")

    thresholds = np.asarray(
        _expanded(decision_threshold, len(samples), "decision_threshold"),
        dtype=float,
    )
    if not np.isfinite(thresholds).all() or (
        ((thresholds < 0.0) | (thresholds > 1.0)).any()
    ):
        raise ValueError("Decision thresholds must be between 0 and 1.")

    ranking = pd.DataFrame(
        {
            "risk_score": scores,
            "predicted_label": (scores >= thresholds).astype(int),
            "true_label": (
                samples[TARGET_COLUMN].astype(int).to_numpy()
                if TARGET_COLUMN in samples
                else pd.array([pd.NA] * len(samples), dtype="Int64")
            ),
            "project": (
                samples[REPOSITORY_COLUMN].astype(str).to_numpy()
                if REPOSITORY_COLUMN in samples
                else [""] * len(samples)
            ),
            "file_path": (
                samples[PATH_COLUMN].astype(str).to_numpy()
                if PATH_COLUMN in samples
                else [""] * len(samples)
            ),
            "snapshot_date": (
                pd.to_datetime(samples[DATE_COLUMN])
                .dt.strftime("%Y-%m-%d")
                .to_numpy()
                if DATE_COLUMN in samples
                else [""] * len(samples)
            ),
            "model": _expanded(model, len(samples), "model"),
            "evaluation_mode": [evaluation_mode] * len(samples),
            "score_method": _expanded(
                score_method, len(samples), "score_method"
            ),
            "decision_threshold": thresholds,
            "_source_order": np.arange(len(samples)),
        }
    )
    ranking = ranking.sort_values(
        [
            "risk_score",
            "project",
            "file_path",
            "snapshot_date",
            "model",
            "_source_order",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking["top_risk_feature"] = pd.NA
    ranking["top_risk_contribution"] = np.nan
    ranking["top_protective_feature"] = pd.NA
    ranking["top_protective_contribution"] = np.nan
    return ranking.loc[:, RISK_RANKING_COLUMNS]


def build_risk_summary(
    ranking: pd.DataFrame,
    *,
    top_risk: int = DEFAULT_TOP_RISK,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    selected_model: str,
    evaluation_mode: str,
) -> dict[str, Any]:
    """Summarize high-risk rows without changing their model predictions."""
    if top_risk <= 0:
        raise ValueError("top_risk must be positive.")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("risk_threshold must be between 0 and 1.")
    if ranking.empty:
        raise ValueError("Risk summaries require at least one evaluated sample.")

    high_risk = ranking.loc[ranking["risk_score"] >= risk_threshold]
    top_columns = [
        "rank",
        "risk_score",
        "predicted_label",
        "true_label",
        "project",
        "file_path",
        "snapshot_date",
        "model",
        "score_method",
    ]
    top_samples = high_risk.head(top_risk)[top_columns].to_dict(orient="records")
    for sample in top_samples:
        if pd.isna(sample["true_label"]):
            sample["true_label"] = None
    return {
        "total_samples": len(ranking),
        "high_risk_sample_count": len(high_risk),
        "risk_threshold": float(risk_threshold),
        "mean_risk_score": float(ranking["risk_score"].mean()),
        "max_risk_score": float(ranking["risk_score"].max()),
        "top_risk_limit": top_risk,
        "top_high_risk_samples": top_samples,
        "model_metadata": {
            "selected_model": selected_model,
            "models_used": sorted(ranking["model"].unique().tolist()),
            "evaluation_mode": evaluation_mode,
            "score_methods": sorted(ranking["score_method"].unique().tolist()),
            "decision_thresholds": sorted(
                ranking["decision_threshold"].unique().tolist()
            ),
        },
    }
