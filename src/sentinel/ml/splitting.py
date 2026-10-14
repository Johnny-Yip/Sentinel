"""Chronological, date-group-preserving dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sentinel.ml.data import DATE_COLUMN, InvalidDatasetError, TARGET_COLUMN, validate_dataset
from sentinel.ml.temporal import partition_mature_data


@dataclass(frozen=True)
class SplitSummary:
    """Human- and machine-readable statistics for one temporal partition."""

    name: str
    start_date: str
    end_date: str
    rows: int
    positives: int
    negatives: int
    positive_percentage: float


@dataclass(frozen=True)
class TemporalSplits:
    """Training, validation, and test partitions in chronological order."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    temporal_context: dict[str, Any] | None = None

    def frames(self) -> tuple[tuple[str, pd.DataFrame], ...]:
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        )

    def summaries(self) -> dict[str, SplitSummary]:
        return {name: summarize_split(name, frame) for name, frame in self.frames()}


def chronological_split(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    *,
    validation_start: str | None = None,
    test_start: str | None = None,
    test_end: str | None = None,
) -> TemporalSplits:
    """Choose date-only candidate boundaries, then purge immature label windows."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave a test period.")

    data = validate_dataset(data)
    dates = data[DATE_COLUMN].drop_duplicates().sort_values().tolist()
    if len(dates) < 3:
        raise InvalidDatasetError(
            "At least three distinct snapshot dates are required for temporal splitting."
        )

    train_date_count = max(1, int(len(dates) * train_fraction))
    validation_end = max(
        train_date_count + 1,
        int(len(dates) * (train_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, len(dates) - 1)

    if (validation_start is None) != (test_start is None):
        raise InvalidDatasetError("Supply both validation_start and test_start, or neither.")
    frames, context = partition_mature_data(
        data,
        validation_start=dates[train_date_count] if validation_start is None else validation_start,
        test_start=dates[validation_end] if test_start is None else test_start,
        test_end=test_end,
    )
    context["boundary_source"] = "distinct_date_fraction_candidates" if validation_start is None else "explicit_calendar_dates"
    context["candidate_fractions"] = {"train": train_fraction, "validation": validation_fraction}
    return TemporalSplits(**frames, temporal_context=context)


def summarize_split(name: str, frame: pd.DataFrame) -> SplitSummary:
    """Calculate date and class statistics for one split."""
    positives = int(frame[TARGET_COLUMN].sum())
    rows = len(frame)
    start = frame[DATE_COLUMN].min().strftime("%Y-%m-%d")
    end = frame[DATE_COLUMN].max().strftime("%Y-%m-%d")
    return SplitSummary(
        name=name,
        start_date=start,
        end_date=end,
        rows=rows,
        positives=positives,
        negatives=rows - positives,
        positive_percentage=positives / rows * 100,
    )
