"""Build leakage-safe monthly file snapshots from mined Java history."""

from __future__ import annotations

import argparse
import re
import sys
from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


HISTORY_COLUMNS = [
    "repository",
    "commit_hash",
    "author_name",
    "author_email",
    "commit_date",
    "commit_message",
    "file_path",
    "lines_added",
    "lines_deleted",
]

FEATURE_COLUMNS = [
    "repository",
    "file_path",
    "snapshot_date",
    "commit_count",
    "developer_count",
    "lines_added",
    "lines_deleted",
    "code_churn",
    "file_age_days",
    "days_since_last_change",
    "previous_bug_fixes",
    "defect_next_90_days",
]

LABEL_WINDOW_DAYS = 90

# Word boundaries prevent broad matches such as "fixture", "debug", and
# "exceptional" while still accepting punctuation such as "fix: null handling".
_BUG_FIX_PATTERN = re.compile(
    r"\b(?:fix|fixed|fixes|bugs?|bugfix(?:es)?|defects?|"
    r"crash(?:es|ed)?|incorrect(?:ly)?|exceptions?|npes?|regressions?)\b",
    re.IGNORECASE,
)


class FeatureEngineeringError(Exception):
    """Base class for expected feature-generation failures."""


class InvalidHistoryError(FeatureEngineeringError):
    """Raised when an input does not match the repository miner schema."""


class NoEligibleSnapshotsError(FeatureEngineeringError):
    """Raised when the history cannot support a complete 90-day label window."""


def is_bug_fix_message(message: object) -> bool:
    """Return whether ``message`` contains a conservative bug-fix term."""
    if not isinstance(message, str):
        return False
    return _BUG_FIX_PATTERN.search(message) is not None


def _validate_and_prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    """Validate miner output and add internal, leakage-safe helper columns."""
    missing_columns = [column for column in HISTORY_COLUMNS if column not in history]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise InvalidHistoryError(f"History CSV is missing required columns: {missing}.")
    if history.empty:
        raise InvalidHistoryError("History CSV contains no Java file changes.")

    data = history.loc[:, HISTORY_COLUMNS].copy()

    for column in ("repository", "commit_hash", "author_name", "file_path"):
        invalid = data[column].isna() | data[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise InvalidHistoryError(f"Column {column!r} contains empty values.")

    non_java_paths = ~data["file_path"].astype(str).str.endswith(".java")
    if non_java_paths.any():
        raise InvalidHistoryError("History CSV contains non-Java file paths.")

    data["commit_date"] = pd.to_datetime(data["commit_date"], utc=True, errors="coerce")
    if data["commit_date"].isna().any():
        raise InvalidHistoryError("Column 'commit_date' contains invalid timestamps.")

    for column in ("lines_added", "lines_deleted"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise InvalidHistoryError(
                f"Column {column!r} must contain non-negative whole numbers."
            )
        data[column] = numeric.astype("int64")

    duplicate_change = data.duplicated(["repository", "commit_hash", "file_path"])
    if duplicate_change.any():
        raise InvalidHistoryError(
            "History CSV contains duplicate repository/commit/file change rows."
        )

    email = data["author_email"].fillna("").astype(str).str.strip().str.lower()
    name = data["author_name"].astype(str).str.strip().str.lower()
    data["_developer"] = email.where(email.ne(""), "name:" + name)
    data["_commit_day"] = data["commit_date"].dt.normalize().dt.tz_localize(None)
    data["_is_bug_fix"] = data["commit_message"].map(is_bug_fix_message)

    return data.sort_values(
        ["repository", "file_path", "commit_date", "commit_hash"],
        kind="stable",
    )


def _month_ends(first_change_day: pd.Timestamp, last_eligible_day: pd.Timestamp):
    """Return month ends from a file's first observable month through the cutoff."""
    first_snapshot = first_change_day + pd.offsets.MonthEnd(0)
    return pd.date_range(
        start=first_snapshot,
        end=last_eligible_day,
        freq=pd.offsets.MonthEnd(),
    )


def build_file_snapshots(history: pd.DataFrame) -> pd.DataFrame:
    """Create cumulative monthly file features and future 90-day defect labels.

    Features use changes on or before each UTC snapshot date. Labels alone inspect
    bug-fix changes strictly after that date and no later than 90 days afterward.
    A repository's latest observed Java change is its observation endpoint, so
    snapshots without a complete label window are omitted.
    """
    data = _validate_and_prepare_history(history)
    observation_end = data.groupby("repository")["_commit_day"].max()
    rows: list[dict[str, object]] = []

    for (repository, file_path), changes in data.groupby(
        ["repository", "file_path"], sort=True
    ):
        changes = changes.reset_index(drop=True)
        first_change_day = changes.loc[0, "_commit_day"]
        last_eligible_day = observation_end.loc[repository] - pd.Timedelta(
            days=LABEL_WINDOW_DAYS
        )
        snapshots = _month_ends(first_change_day, last_eligible_day)
        if snapshots.empty:
            continue

        event_days = changes["_commit_day"].tolist()
        bug_fix_days = changes.loc[changes["_is_bug_fix"], "_commit_day"].tolist()
        developers: set[str] = set()
        commits: set[str] = set()
        prior_bug_fix_commits: set[str] = set()
        cumulative_added = 0
        cumulative_deleted = 0
        event_index = 0
        last_change_day = first_change_day

        for snapshot in snapshots:
            while event_index < len(changes) and event_days[event_index] <= snapshot:
                change = changes.iloc[event_index]
                commit_hash = str(change["commit_hash"])
                commits.add(commit_hash)
                developers.add(str(change["_developer"]))
                cumulative_added += int(change["lines_added"])
                cumulative_deleted += int(change["lines_deleted"])
                if bool(change["_is_bug_fix"]):
                    prior_bug_fix_commits.add(commit_hash)
                last_change_day = change["_commit_day"]
                event_index += 1

            future_start = bisect_right(bug_fix_days, snapshot)
            future_end = bisect_right(
                bug_fix_days, snapshot + pd.Timedelta(days=LABEL_WINDOW_DAYS)
            )

            rows.append(
                {
                    "repository": repository,
                    "file_path": file_path,
                    "snapshot_date": snapshot.strftime("%Y-%m-%d"),
                    "commit_count": len(commits),
                    "developer_count": len(developers),
                    "lines_added": cumulative_added,
                    "lines_deleted": cumulative_deleted,
                    "code_churn": cumulative_added + cumulative_deleted,
                    "file_age_days": (snapshot - first_change_day).days,
                    "days_since_last_change": (snapshot - last_change_day).days,
                    "previous_bug_fixes": len(prior_bug_fix_commits),
                    "defect_next_90_days": int(future_end > future_start),
                }
            )

    if not rows:
        raise NoEligibleSnapshotsError(
            "History does not contain a file with a complete 90-day label window."
        )

    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def default_output_path(input_path: str | Path) -> Path:
    """Return the default V2 CSV path for a V1 history CSV."""
    source = Path(input_path)
    base_name = source.stem
    if base_name.endswith("_java_history"):
        base_name = base_name.removesuffix("_java_history")
    return source.with_name(f"{base_name}_features.csv")


def export_features(data: pd.DataFrame, output_path: str | Path) -> Path:
    """Export file snapshots with a stable column order."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False, columns=FEATURE_COLUMNS)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe monthly Java file snapshots from mined history."
    )
    parser.add_argument("history_csv", type=Path, help="CSV generated by sentinel.mine")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="CSV destination (default: <repository>_features.csv)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or default_output_path(args.history_csv)

    try:
        source = args.history_csv.expanduser().resolve()
        if source == output_path.expanduser().resolve():
            raise FeatureEngineeringError("Input and output CSV paths must differ.")
        history = pd.read_csv(source)
        features = build_file_snapshots(history)
        destination = export_features(features, output_path)
    except (
        FeatureEngineeringError,
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    positive_labels = int(features["defect_next_90_days"].sum())
    negative_labels = len(features) - positive_labels
    positive_percentage = positive_labels / len(features) * 100

    print(f"Generated {len(features):,} monthly file snapshots.")
    unique_files = features[["repository", "file_path"]].drop_duplicates().shape[0]
    print(f"Unique Java files: {unique_files:,}")
    print(f"Positive labels: {positive_labels:,}")
    print(f"Negative labels: {negative_labels:,}")
    print(f"Positive-label percentage: {positive_percentage:.2f}%")
    print(f"CSV written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
