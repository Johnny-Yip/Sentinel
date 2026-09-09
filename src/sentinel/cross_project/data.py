"""Load and summarize multiple leakage-safe V2 feature datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sentinel.ml.data import (
    DATE_COLUMN,
    MODEL_FEATURE_COLUMNS,
    PATH_COLUMN,
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
    InvalidDatasetError,
    load_dataset,
    validate_dataset,
)


@dataclass(frozen=True)
class RepositorySummary:
    """Dataset statistics for one repository."""

    repository: str
    snapshot_count: int
    unique_java_files: int
    positive_labels: int
    negative_labels: int
    positive_rate: float
    start_date: str
    end_date: str
    feature_csv_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MultiRepositoryDataset:
    """Validated snapshots plus their source-level summaries."""

    data: pd.DataFrame
    summaries: tuple[RepositorySummary, ...]
    input_paths: tuple[Path, ...]
    feature_columns: tuple[str, ...]


def _repository_summary(
    repository: str, frame: pd.DataFrame, source: Path
) -> RepositorySummary:
    positives = int(frame[TARGET_COLUMN].sum())
    snapshots = len(frame)
    return RepositorySummary(
        repository=repository,
        snapshot_count=snapshots,
        unique_java_files=int(frame[PATH_COLUMN].nunique()),
        positive_labels=positives,
        negative_labels=snapshots - positives,
        positive_rate=positives / snapshots,
        start_date=frame[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        end_date=frame[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        feature_csv_path=str(source),
    )


def load_multi_repository_datasets(
    feature_csvs: list[str | Path] | tuple[str | Path, ...],
) -> MultiRepositoryDataset:
    """Load datasets, enforce a common predictor schema, and combine rows.

    A CSV may contain one or more repositories, but a repository must not be
    spread across input files. This keeps source reporting unambiguous and
    catches accidental duplicate inputs early.
    """
    if not feature_csvs:
        raise InvalidDatasetError(
            "Cross-project evaluation requires at least one feature CSV input."
        )

    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    source_by_repository: dict[str, Path] = {}

    for feature_csv in feature_csvs:
        source = Path(feature_csv).expanduser().resolve()
        data = load_dataset(source)
        repositories = sorted(data[REPOSITORY_COLUMN].unique().tolist())
        for repository in repositories:
            if repository in source_by_repository:
                previous = source_by_repository[repository]
                raise InvalidDatasetError(
                    f"Repository {repository!r} appears in more than one input CSV: "
                    f"{previous} and {source}."
                )
            source_by_repository[repository] = source
        frames.append(data)
        paths.append(source)

    combined = validate_dataset(pd.concat(frames, ignore_index=True, sort=False))
    repositories = sorted(combined[REPOSITORY_COLUMN].unique().tolist())
    if len(repositories) < 2:
        raise InvalidDatasetError(
            "Cross-project evaluation requires snapshots from at least two "
            "distinct repositories."
        )

    summaries: list[RepositorySummary] = []
    for repository in repositories:
        repository_rows = combined.loc[
            combined[REPOSITORY_COLUMN] == repository
        ]
        if repository_rows[DATE_COLUMN].nunique() < 2:
            raise InvalidDatasetError(
                f"Repository {repository!r} needs at least two distinct snapshot "
                "dates for chronological training/validation splitting."
            )
        summaries.append(
            _repository_summary(
                repository, repository_rows, source_by_repository[repository]
            )
        )

    # The predictor whitelist is the V2 schema contract. Unknown columns may be
    # retained as metadata, but never enter this tuple or any sklearn pipeline.
    feature_columns = tuple(MODEL_FEATURE_COLUMNS)
    return MultiRepositoryDataset(
        data=combined,
        summaries=tuple(summaries),
        input_paths=tuple(paths),
        feature_columns=feature_columns,
    )
