from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sentinel.features import (
    FEATURE_COLUMNS,
    build_file_snapshots,
    export_features,
    is_bug_fix_message,
    main,
)
from sentinel.miner import CSV_COLUMNS


def history_row(
    *,
    commit_hash: str,
    commit_date: str,
    file_path: str,
    message: str,
    author_name: str = "Alice",
    author_email: str = "alice@example.com",
    lines_added: int = 1,
    lines_deleted: int = 0,
) -> dict[str, object]:
    return {
        "repository": "owner/repository",
        "commit_hash": commit_hash,
        "author_name": author_name,
        "author_email": author_email,
        "commit_date": commit_date,
        "commit_message": message,
        "file_path": file_path,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }


def make_history(include_future_bug_fix: bool = True) -> pd.DataFrame:
    rows = [
        history_row(
            commit_hash="a1",
            commit_date="2024-01-05T10:00:00+00:00",
            file_path="src/A.java",
            message="Add A",
            lines_added=10,
            lines_deleted=2,
        ),
        history_row(
            commit_hash="a2",
            commit_date="2024-01-20T10:00:00+00:00",
            file_path="src/A.java",
            message="Refactor A",
            author_name="Bob",
            author_email="bob@example.com",
            lines_added=3,
            lines_deleted=4,
        ),
        history_row(
            commit_hash="b1",
            commit_date="2024-01-10T10:00:00+00:00",
            file_path="src/NoFix.java",
            message="Add another class",
            lines_added=5,
            lines_deleted=1,
        ),
        history_row(
            commit_hash="b2",
            commit_date="2024-03-15T10:00:00+00:00",
            file_path="src/NoFix.java",
            message="Improve documentation",
            lines_added=1,
            lines_deleted=1,
        ),
        # Extends the repository observation period beyond the January label window.
        history_row(
            commit_hash="z1",
            commit_date="2024-05-01T10:00:00+00:00",
            file_path="src/ObservationMarker.java",
            message="Update marker",
        ),
    ]
    if include_future_bug_fix:
        rows.append(
            history_row(
                commit_hash="a3",
                commit_date="2024-02-15T10:00:00+00:00",
                file_path="src/A.java",
                message="Fix NPE in A",
                author_name="Carol",
                author_email="carol@example.com",
                lines_added=100,
                lines_deleted=50,
            )
        )
    return pd.DataFrame(rows, columns=CSV_COLUMNS)


@pytest.mark.parametrize(
    "message",
    [
        "fix null handling",
        "Fixed parser state",
        "This fixes LANG-123",
        "bug in tokenizer",
        "bugfix: retain escapes",
        "Resolve defect",
        "Prevent crash",
        "Incorrect output for empty input",
        "Handle exception",
        "Avoid NPE",
        "Regression in builder",
    ],
)
def test_bug_fix_heuristic_matches_conservative_terms(message: str) -> None:
    assert is_bug_fix_message(message)


@pytest.mark.parametrize(
    "message",
    [
        "Add a new feature",
        "Refactor fixture setup",
        "Add debug logging",
        "Exceptional performance improvement",
        "Update documentation",
        "prefix validation rules",
        None,
    ],
)
def test_bug_fix_heuristic_avoids_broad_matches(message: object) -> None:
    assert not is_bug_fix_message(message)


def test_feature_calculations_include_developers_and_churn() -> None:
    result = build_file_snapshots(make_history())
    snapshot = result.loc[
        (result["file_path"] == "src/A.java")
        & (result["snapshot_date"] == "2024-01-31")
    ].squeeze()

    assert list(result.columns) == FEATURE_COLUMNS
    assert snapshot["repository"] == "owner/repository"
    assert snapshot["commit_count"] == 2
    assert snapshot["developer_count"] == 2
    assert snapshot["lines_added"] == 13
    assert snapshot["lines_deleted"] == 6
    assert snapshot["code_churn"] == 19
    assert snapshot["file_age_days"] == 26
    assert snapshot["days_since_last_change"] == 11
    assert snapshot["previous_bug_fixes"] == 0


def test_future_bug_fix_sets_90_day_label() -> None:
    result = build_file_snapshots(make_history())
    snapshot = result.loc[
        (result["file_path"] == "src/A.java")
        & (result["snapshot_date"] == "2024-01-31")
    ].squeeze()

    assert snapshot["defect_next_90_days"] == 1


def test_file_without_future_bug_fix_has_negative_label() -> None:
    result = build_file_snapshots(make_history())
    snapshot = result.loc[
        (result["file_path"] == "src/NoFix.java")
        & (result["snapshot_date"] == "2024-01-31")
    ].squeeze()

    assert snapshot["defect_next_90_days"] == 0


def test_file_receives_each_eligible_month_end_snapshot() -> None:
    history = make_history()
    history.loc[history["commit_hash"] == "z1", "commit_date"] = (
        "2024-06-30T10:00:00+00:00"
    )

    result = build_file_snapshots(history)
    snapshot_dates = result.loc[result["file_path"] == "src/A.java", "snapshot_date"]

    assert snapshot_dates.tolist() == ["2024-01-31", "2024-02-29", "2024-03-31"]


def test_future_changes_do_not_leak_into_features() -> None:
    with_future_fix = build_file_snapshots(make_history(include_future_bug_fix=True))
    without_future_fix = build_file_snapshots(make_history(include_future_bug_fix=False))

    selector = lambda frame: frame.loc[
        (frame["file_path"] == "src/A.java")
        & (frame["snapshot_date"] == "2024-01-31")
    ].reset_index(drop=True)
    future_snapshot = selector(with_future_fix)
    baseline_snapshot = selector(without_future_fix)

    feature_columns = [
        column for column in FEATURE_COLUMNS if column != "defect_next_90_days"
    ]
    pd.testing.assert_frame_equal(
        future_snapshot[feature_columns], baseline_snapshot[feature_columns]
    )
    assert future_snapshot.loc[0, "defect_next_90_days"] == 1
    assert baseline_snapshot.loc[0, "defect_next_90_days"] == 0


def test_label_window_is_strictly_future_and_includes_day_90() -> None:
    history = pd.DataFrame(
        [
            history_row(
                commit_hash="base",
                commit_date="2024-01-01T10:00:00+00:00",
                file_path="src/Boundary.java",
                message="Add boundary case",
            ),
            history_row(
                commit_hash="same-day",
                commit_date="2024-01-31T10:00:00+00:00",
                file_path="src/Boundary.java",
                message="Fix same-day issue",
            ),
            history_row(
                commit_hash="day-90",
                commit_date="2024-04-30T10:00:00+00:00",
                file_path="src/Boundary.java",
                message="Fix boundary regression",
            ),
            history_row(
                commit_hash="late-base",
                commit_date="2024-01-01T10:00:00+00:00",
                file_path="src/Late.java",
                message="Add late case",
            ),
            history_row(
                commit_hash="day-91",
                commit_date="2024-05-01T10:00:00+00:00",
                file_path="src/Late.java",
                message="Fix after the label window",
            ),
            history_row(
                commit_hash="observe",
                commit_date="2024-05-02T10:00:00+00:00",
                file_path="src/ObservationMarker.java",
                message="Update marker",
            ),
        ],
        columns=CSV_COLUMNS,
    )

    result = build_file_snapshots(history)
    snapshot = result.loc[
        (result["file_path"] == "src/Boundary.java")
        & (result["snapshot_date"] == "2024-01-31")
    ].squeeze()

    assert snapshot["previous_bug_fixes"] == 1
    assert snapshot["defect_next_90_days"] == 1

    late_snapshot = result.loc[
        (result["file_path"] == "src/Late.java")
        & (result["snapshot_date"] == "2024-01-31")
    ].squeeze()
    assert late_snapshot["defect_next_90_days"] == 0


def test_feature_cli_exports_csv_and_prints_summary(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "example_java_history.csv"
    destination = tmp_path / "example_features.csv"
    make_history().to_csv(source, index=False)

    exit_code = main([str(source), "--output", str(destination)])

    captured = capsys.readouterr()
    exported = pd.read_csv(destination)
    assert exit_code == 0
    assert list(exported.columns) == FEATURE_COLUMNS
    assert "monthly file snapshots" in captured.out
    assert "Unique Java files:" in captured.out
    assert "Positive labels:" in captured.out
    assert "Negative labels:" in captured.out
    assert "Positive-label percentage:" in captured.out
    assert captured.err == ""


def test_export_features_preserves_column_order(tmp_path: Path) -> None:
    result = build_file_snapshots(make_history())

    destination = export_features(result, tmp_path / "nested" / "features.csv")
    exported = pd.read_csv(destination)

    assert list(exported.columns) == FEATURE_COLUMNS
