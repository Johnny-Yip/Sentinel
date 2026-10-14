"""Leave-one-project-out folds with per-repository chronological validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from sentinel.ml.data import DATE_COLUMN, REPOSITORY_COLUMN, InvalidDatasetError
from sentinel.ml.splitting import TemporalSplits
from sentinel.ml.temporal import partition_mature_data


@dataclass(frozen=True)
class CrossProjectFold:
    """One repository-disjoint train/validation/test fold."""

    held_out_repository: str
    training_repositories: tuple[str, ...]
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    repository_date_ranges: dict[str, dict[str, str]]
    temporal_context: dict[str, Any] | None = None

    def as_temporal_splits(self) -> TemporalSplits:
        return TemporalSplits(
            train=self.train,
            validation=self.validation,
            test=self.test,
            temporal_context=self.temporal_context,
        )



def build_leave_one_project_out_folds(
    data: pd.DataFrame, train_fraction: float | None = None, *,
    validation_start: str | None = None, test_start: str | None = None,
    test_end: str | None = None,
) -> tuple[CrossProjectFold, ...]:
    """Repository-disjoint forward folds on one frozen historical schedule."""
    if train_fraction is not None:
        raise InvalidDatasetError("Independent train_fraction splits are legacy/unverified; supply shared calendar boundaries.")
    if validation_start is None or test_start is None:
        raise InvalidDatasetError("Prospective LOPO requires explicit validation_start and test_start shared across repositories.")
    repositories = sorted(data[REPOSITORY_COLUMN].unique().tolist())
    if len(repositories) < 2:
        raise InvalidDatasetError("Leave-one-project-out evaluation requires at least two repositories.")
    folds = []
    for held_out in repositories:
        frames, context = partition_mature_data(
            data, validation_start=validation_start, test_start=test_start,
            test_end=test_end, held_out_repository=held_out,
        )
        context["boundary_source"] = "explicit_shared_calendar_dates"
        sources = tuple(name for name in repositories if name != held_out)
        ranges = {}
        for repository in sources:
            train = frames["train"].loc[frames["train"][REPOSITORY_COLUMN] == repository]
            validation = frames["validation"].loc[frames["validation"][REPOSITORY_COLUMN] == repository]
            ranges[repository] = {
                "training_start": train[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "training_end": train[DATE_COLUMN].max().strftime("%Y-%m-%d"),
                "validation_start": validation[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                "validation_end": validation[DATE_COLUMN].max().strftime("%Y-%m-%d"),
            }
        folds.append(CrossProjectFold(
            held_out_repository=held_out, training_repositories=sources,
            **frames, repository_date_ranges=ranges, temporal_context=context,
        ))
    return tuple(folds)
