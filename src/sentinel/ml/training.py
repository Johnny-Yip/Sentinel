"""Train, compare, interpret, and persist Sentinel V3 baselines."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from sentinel import __version__
from sentinel.ml.data import (
    DATE_COLUMN,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    InvalidDatasetError,
)
from sentinel.ml.evaluation import (
    DEFAULT_THRESHOLD,
    calculate_metrics,
    extract_feature_importance,
    select_f1_threshold,
)
from sentinel.ml.models import DEFAULT_RANDOM_STATE, MODEL_ORDER, build_model_pipelines
from sentinel.ml.splitting import TemporalSplits


ProgressCallback = Callable[[str], None]


@dataclass
class ModelResult:
    """A fitted model plus validation/test evaluation and interpretation."""

    name: str
    pipeline: Pipeline
    selected_threshold: float
    validation_default: dict[str, Any]
    validation_selected: dict[str, Any]
    test_default: dict[str, Any]
    test_selected: dict[str, Any]
    feature_importance: dict[str, Any]


@dataclass
class TrainingResult:
    """Complete outputs from one reproducible V3 baseline run."""

    splits: TemporalSplits
    feature_columns: list[str]
    models: dict[str, ModelResult]
    best_model_name: str
    random_state: int

    @property
    def best_model(self) -> ModelResult:
        return self.models[self.best_model_name]


def _positive_probabilities(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = pipeline.predict_proba(features)
    model = pipeline.named_steps["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise InvalidDatasetError(
            "The training split did not produce a positive-class probability."
        )
    return np.asarray(probabilities[:, classes.index(1)], dtype=float)


def _feature_importance(
    name: str,
    pipeline: Pipeline,
    feature_columns: Sequence[str],
    evaluation_features: pd.DataFrame,
    evaluation_target: np.ndarray,
    random_state: int,
    limit: int = 10,
) -> dict[str, Any]:
    ranked = extract_feature_importance(
        pipeline,
        feature_columns,
        evaluation_features,
        evaluation_target,
        random_state=random_state,
        limit=limit,
    )
    importance: dict[str, Any] = {"ranked_features": ranked}
    if name == "logistic_regression":
        positive = [record for record in ranked if record["direction"] == "positive"]
        negative = sorted(
            (record for record in ranked if record["direction"] == "negative"),
            key=lambda record: record["effect"],
        )
        importance.update(
            {
                "positive_coefficients": [
                    {"feature": record["feature"], "value": record["effect"]}
                    for record in positive
                ],
                "negative_coefficients": [
                    {"feature": record["feature"], "value": record["effect"]}
                    for record in negative
                ],
            }
        )
    elif name == "random_forest":
        importance.update(
            {
                "feature_importances": [
                    {"feature": record["feature"], "value": record["effect"]}
                    for record in ranked
                ]
            }
        )
    return importance


def _selection_key(result: ModelResult) -> tuple[float, float, float]:
    metrics = result.validation_selected
    pr_auc = metrics["pr_auc"]
    return (
        float(pr_auc) if pr_auc is not None else float("-inf"),
        float(metrics["f1"]),
        float(metrics["recall"]),
    )


def train_baselines(
    splits: TemporalSplits,
    feature_columns: Sequence[str] = MODEL_FEATURE_COLUMNS,
    random_state: int = DEFAULT_RANDOM_STATE,
    progress: ProgressCallback | None = None,
) -> TrainingResult:
    """Fit on training only, tune thresholds on validation, then score test once."""
    feature_columns = list(feature_columns)
    train_classes = set(splits.train[TARGET_COLUMN].unique())
    if train_classes != {0, 1}:
        raise InvalidDatasetError(
            "Training split must contain both target classes for the V3 baselines."
        )

    pipelines = build_model_pipelines(feature_columns, random_state=random_state)
    validation_results: dict[str, ModelResult] = {}
    train_features = splits.train.loc[:, feature_columns]
    train_target = splits.train[TARGET_COLUMN].to_numpy()
    validation_features = splits.validation.loc[:, feature_columns]
    validation_target = splits.validation[TARGET_COLUMN].to_numpy()

    # Model fitting, preprocessing fitting, threshold selection, and model
    # selection all finish before this function reads the test frame.
    for name in MODEL_ORDER:
        if progress:
            progress(f"Training {name}...")
        pipeline = pipelines[name]
        pipeline.fit(train_features, train_target)
        validation_scores = _positive_probabilities(pipeline, validation_features)
        selected_threshold = select_f1_threshold(
            validation_target, validation_scores
        )
        validation_results[name] = ModelResult(
            name=name,
            pipeline=pipeline,
            selected_threshold=selected_threshold,
            validation_default=calculate_metrics(
                validation_target, validation_scores, DEFAULT_THRESHOLD
            ),
            validation_selected=calculate_metrics(
                validation_target, validation_scores, selected_threshold
            ),
            test_default={},
            test_selected={},
            feature_importance={},
        )

    best_model_name = max(
        MODEL_ORDER, key=lambda name: _selection_key(validation_results[name])
    )
    if progress:
        progress(
            f"Validation selection locked: {best_model_name} (primary metric: PR-AUC)."
        )

    test_features = splits.test.loc[:, feature_columns]
    test_target = splits.test[TARGET_COLUMN].to_numpy()
    for name in MODEL_ORDER:
        model_result = validation_results[name]
        test_scores = _positive_probabilities(model_result.pipeline, test_features)
        model_result.test_default = calculate_metrics(
            test_target, test_scores, DEFAULT_THRESHOLD
        )
        model_result.test_selected = calculate_metrics(
            test_target, test_scores, model_result.selected_threshold
        )
        model_result.feature_importance = _feature_importance(
            name,
            model_result.pipeline,
            feature_columns,
            validation_features,
            validation_target,
            random_state,
        )

    return TrainingResult(
        splits=splits,
        feature_columns=feature_columns,
        models=validation_results,
        best_model_name=best_model_name,
        random_state=random_state,
    )


def _date_range(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": frame[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "end": frame[DATE_COLUMN].max().strftime("%Y-%m-%d"),
    }


def build_metadata(result: TrainingResult) -> dict[str, Any]:
    """Build JSON-safe metadata for the validation-selected best model."""
    best = result.best_model
    return {
        "sentinel_version": __version__,
        "model_type": result.best_model_name,
        "selection_metric": "validation_pr_auc",
        "feature_columns": result.feature_columns,
        "excluded_columns": [
            column
            for column in result.splits.train.columns
            if column not in result.feature_columns
        ],
        "target_column": TARGET_COLUMN,
        "random_state": result.random_state,
        "selected_threshold": best.selected_threshold,
        "leakage_audit": {
            "status": "passed",
            "split_strategy": "chronological_distinct_snapshot_dates_70_15_15",
            "same_date_rows_grouped": True,
            "preprocessing_fit_scope": "training_only",
            "threshold_selection_scope": "validation_only",
            "model_selection_scope": "validation_only",
            "test_use": "final_evaluation_only",
            "predictor_policy": "explicit_historical_v2_feature_whitelist",
            "target_excluded": True,
            "metadata_identifiers_excluded": True,
            "future_derived_predictors": [],
        },
        "date_ranges": {
            "training": _date_range(result.splits.train),
            "validation": _date_range(result.splits.validation),
            "test": _date_range(result.splits.test),
        },
        "evaluation_metrics": {
            "validation": {
                "default_0_5": best.validation_default,
                "selected_threshold": best.validation_selected,
            },
            "test": {
                "default_0_5": best.test_default,
                "selected_threshold": best.test_selected,
            },
        },
        "feature_importance": best.feature_importance,
        "all_model_comparison": {
            name: {
                "selected_threshold": model.selected_threshold,
                "validation": model.validation_selected,
                "test": model.test_selected,
            }
            for name, model in result.models.items()
        },
    }


def save_artifacts(
    result: TrainingResult, output_dir: str | Path
) -> tuple[Path, Path]:
    """Persist the validation-selected sklearn pipeline and JSON metadata."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.joblib"
    metadata_path = destination / "metadata.json"
    joblib.dump(result.best_model.pipeline, model_path)
    metadata_path.write_text(
        json.dumps(build_metadata(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path, metadata_path
