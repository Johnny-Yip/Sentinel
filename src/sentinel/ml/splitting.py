"""Chronological, date-group-preserving dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sentinel.ml.data import DATE_COLUMN, InvalidDatasetError, TARGET_COLUMN


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
) -> TemporalSplits:
    """Split whole snapshot dates into earliest 70%, next 15%, final 15%."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Training and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Training and validation fractions must leave a test period.")

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

    train_dates = set(dates[:train_date_count])
    validation_dates = set(dates[train_date_count:validation_end])
    test_dates = set(dates[validation_end:])

    train = data.loc[data[DATE_COLUMN].isin(train_dates)].copy()
    validation = data.loc[data[DATE_COLUMN].isin(validation_dates)].copy()
    test = data.loc[data[DATE_COLUMN].isin(test_dates)].copy()

    if train.empty or validation.empty or test.empty:
        raise InvalidDatasetError(
            "Chronological splitting produced an empty partition; more dates are required."
        )

    return TemporalSplits(train=train, validation=validation, test=test)


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
