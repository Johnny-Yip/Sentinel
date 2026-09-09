"""Leave-one-project-out folds with per-repository chronological validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sentinel.ml.data import DATE_COLUMN, REPOSITORY_COLUMN, InvalidDatasetError
from sentinel.ml.splitting import TemporalSplits


@dataclass(frozen=True)
class CrossProjectFold:
    """One repository-disjoint train/validation/test fold."""

    held_out_repository: str
    training_repositories: tuple[str, ...]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    repository_date_ranges: dict[str, dict[str, str]]

    def as_temporal_splits(self) -> TemporalSplits:
        return TemporalSplits(
            train=self.train,
            validation=self.validation,
            test=self.test,
        )


def _chronological_train_validation(
    frame: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    dates = frame[DATE_COLUMN].drop_duplicates().sort_values().tolist()
    if len(dates) < 2:
        repository = frame[REPOSITORY_COLUMN].iloc[0]
        raise InvalidDatasetError(
            f"Repository {repository!r} needs at least two snapshot dates for "
            "chronological training/validation splitting."
        )

    train_date_count = min(max(1, int(len(dates) * train_fraction)), len(dates) - 1)
    train_dates = set(dates[:train_date_count])
    validation_dates = set(dates[train_date_count:])
    train = frame.loc[frame[DATE_COLUMN].isin(train_dates)].copy()
    validation = frame.loc[frame[DATE_COLUMN].isin(validation_dates)].copy()
    return train, validation, {
        "training_start": min(train_dates).strftime("%Y-%m-%d"),
        "training_end": max(train_dates).strftime("%Y-%m-%d"),
        "validation_start": min(validation_dates).strftime("%Y-%m-%d"),
        "validation_end": max(validation_dates).strftime("%Y-%m-%d"),
    }


def build_leave_one_project_out_folds(
    data: pd.DataFrame, train_fraction: float = 0.80
) -> tuple[CrossProjectFold, ...]:
    """Build one fold per repository, preserving whole dates within each project."""
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("Training fraction must be strictly between 0 and 1.")

    repositories = sorted(data[REPOSITORY_COLUMN].unique().tolist())
    if len(repositories) < 2:
        raise InvalidDatasetError(
            "Leave-one-project-out evaluation requires at least two repositories."
        )

    folds: list[CrossProjectFold] = []
    for held_out in repositories:
        training_repositories = tuple(
            repository for repository in repositories if repository != held_out
        )
        train_parts: list[pd.DataFrame] = []
        validation_parts: list[pd.DataFrame] = []
        date_ranges: dict[str, dict[str, str]] = {}

        for repository in training_repositories:
            repository_rows = data.loc[data[REPOSITORY_COLUMN] == repository]
            train, validation, ranges = _chronological_train_validation(
                repository_rows, train_fraction
            )
            train_parts.append(train)
            validation_parts.append(validation)
            date_ranges[repository] = ranges

        test = data.loc[data[REPOSITORY_COLUMN] == held_out].copy()
        train = pd.concat(train_parts, ignore_index=True).sort_values(
            [DATE_COLUMN, REPOSITORY_COLUMN], kind="stable"
        )
        validation = pd.concat(validation_parts, ignore_index=True).sort_values(
            [DATE_COLUMN, REPOSITORY_COLUMN], kind="stable"
        )
        folds.append(
            CrossProjectFold(
                held_out_repository=held_out,
                training_repositories=training_repositories,
                train=train.reset_index(drop=True),
                validation=validation.reset_index(drop=True),
                test=test.reset_index(drop=True),
                repository_date_ranges=date_ranges,
            )
        )

    return tuple(folds)
