"""Sentinel V3 machine-learning baselines and temporal evaluation."""

from sentinel.ml.data import (
    DATE_COLUMN,
    METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    InvalidDatasetError,
    MLError,
    load_dataset,
    validate_dataset,
)
from sentinel.ml.evaluation import calculate_metrics, select_f1_threshold
from sentinel.ml.splitting import TemporalSplits, chronological_split
from sentinel.ml.training import TrainingResult, save_artifacts, train_baselines

__all__ = [
    "DATE_COLUMN",
    "METADATA_COLUMNS",
    "MODEL_FEATURE_COLUMNS",
    "REQUIRED_COLUMNS",
    "TARGET_COLUMN",
    "InvalidDatasetError",
    "MLError",
    "TemporalSplits",
    "TrainingResult",
    "calculate_metrics",
    "chronological_split",
    "load_dataset",
    "save_artifacts",
    "select_f1_threshold",
    "train_baselines",
    "validate_dataset",
]
