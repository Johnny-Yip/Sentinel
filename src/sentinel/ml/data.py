"""Load and validate leakage-safe V2 feature datasets for Sentinel V3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "defect_next_90_days"
DATE_COLUMN = "snapshot_date"
PATH_COLUMN = "file_path"
REPOSITORY_COLUMN = "repository"

# V3 deliberately trains only on the known historical features produced by V2.
# The whitelist prevents newly added identifiers or future-derived columns from
# silently becoming predictors.
MODEL_FEATURE_COLUMNS = [
    "commit_count",
    "developer_count",
    "lines_added",
    "lines_deleted",
    "code_churn",
    "file_age_days",
    "days_since_last_change",
    "previous_bug_fixes",
]

METADATA_COLUMNS = [REPOSITORY_COLUMN, PATH_COLUMN, DATE_COLUMN]
REQUIRED_COLUMNS = [
    *METADATA_COLUMNS,
    *MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
]


class MLError(Exception):
    """Base class for expected Sentinel V3 failures."""


class InvalidDatasetError(MLError):
    """Raised when a feature dataset cannot safely be used for training."""


def validate_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Validate a V2 feature frame and return a typed, defensive copy."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataset]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise InvalidDatasetError(
            f"Feature CSV is missing required columns: {missing}."
        )
    if dataset.empty:
        raise InvalidDatasetError("Feature CSV contains no file snapshots.")

    data = dataset.copy()
    required_missing = data[REQUIRED_COLUMNS].isna().sum()
    required_missing = required_missing[required_missing > 0]
    if not required_missing.empty:
        details = ", ".join(
            f"{column} ({int(count)})"
            for column, count in required_missing.items()
        )
        raise InvalidDatasetError(
            f"Required columns contain missing values: {details}."
        )

    for column in (REPOSITORY_COLUMN, PATH_COLUMN):
        empty = data[column].astype(str).str.strip().eq("")
        if empty.any():
            raise InvalidDatasetError(f"Column {column!r} contains empty values.")

    parsed_dates = pd.to_datetime(data[DATE_COLUMN], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        invalid_count = int(parsed_dates.isna().sum())
        raise InvalidDatasetError(
            f"Column {DATE_COLUMN!r} contains {invalid_count} invalid date value(s)."
        )
    data[DATE_COLUMN] = parsed_dates.dt.tz_localize(None).dt.normalize()

    for column in MODEL_FEATURE_COLUMNS:
        numeric = pd.to_numeric(data[column], errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric)
        if invalid.any():
            raise InvalidDatasetError(
                f"Feature column {column!r} must contain only finite numeric values."
            )
        if (numeric < 0).any():
            raise InvalidDatasetError(
                f"Feature column {column!r} must contain non-negative values."
            )
        data[column] = numeric.astype("float64")

    target = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise InvalidDatasetError(
            f"Target column {TARGET_COLUMN!r} must contain only binary 0/1 values."
        )
    data[TARGET_COLUMN] = target.astype("int8")

    duplicates = data.duplicated(
        [REPOSITORY_COLUMN, PATH_COLUMN, DATE_COLUMN], keep=False
    )
    if duplicates.any():
        raise InvalidDatasetError(
            "Feature CSV contains duplicate repository/file/snapshot rows."
        )

    if data[DATE_COLUMN].nunique() < 3:
        raise InvalidDatasetError(
            "Feature CSV must contain at least three distinct snapshot dates for "
            "chronological train/validation/test splitting."
        )

    return data.sort_values(
        [DATE_COLUMN, REPOSITORY_COLUMN, PATH_COLUMN], kind="stable"
    ).reset_index(drop=True)


def load_dataset(feature_csv: str | Path) -> pd.DataFrame:
    """Load and validate a V2 feature CSV."""
    source = Path(feature_csv).expanduser().resolve()
    try:
        dataset = pd.read_csv(source)
    except FileNotFoundError as exc:
        raise InvalidDatasetError(f"Feature CSV does not exist: {source}") from exc
    except pd.errors.EmptyDataError as exc:
        raise InvalidDatasetError(f"Feature CSV is empty: {source}") from exc
    except pd.errors.ParserError as exc:
        raise InvalidDatasetError(f"Feature CSV could not be parsed: {exc}") from exc
    return validate_dataset(dataset)
