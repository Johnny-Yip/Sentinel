"""Centralized, readable configuration for Sentinel V3 baseline models."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_STATE = 42
RANDOM_FOREST_TREES = 200
LOGISTIC_MAX_ITERATIONS = 1_000
MODEL_ORDER = ("dummy", "logistic_regression", "random_forest")


def _preprocessor(feature_columns: Sequence[str], *, scale: bool) -> ColumnTransformer:
    transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        [("numeric", transformer, list(feature_columns))],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipelines(
    feature_columns: Sequence[str], random_state: int = DEFAULT_RANDOM_STATE
) -> dict[str, Pipeline]:
    """Build fresh sklearn pipelines for all supported V3 baselines."""
    return {
        "dummy": Pipeline(
            [
                ("preprocess", _preprocessor(feature_columns, scale=False)),
                ("model", DummyClassifier(strategy="prior")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor(feature_columns, scale=True)),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=LOGISTIC_MAX_ITERATIONS,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _preprocessor(feature_columns, scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=RANDOM_FOREST_TREES,
                        class_weight="balanced_subsample",
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
