from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.evaluation.cli import main
from sentinel.evaluation.experiment import (
    EvaluationReport,
    evaluate_cross_project,
)
from sentinel.evaluation.reporting import REPORT_FILENAMES, save_evaluation_report


def make_feature_data(repository: str, *, offset: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2021-01-31", periods=12, freq="ME")
    for date_index, snapshot_date in enumerate(dates):
        for file_index in range(4):
            commit_count = date_index + file_index + offset + 1
            lines_added = commit_count * 6
            lines_deleted = commit_count * 2
            rows.append(
                {
                    "repository": repository,
                    "file_path": f"src/File{file_index}.java",
                    "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                    "commit_count": commit_count,
                    "developer_count": file_index + 1,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "code_churn": lines_added + lines_deleted,
                    "file_age_days": date_index * 30 + file_index,
                    "days_since_last_change": (date_index + file_index) % 19,
                    "previous_bug_fixes": (date_index + file_index + offset) % 3,
                    "defect_next_90_days": int(file_index == 0),
                }
            )
    return pd.DataFrame(rows)


def sample_report() -> EvaluationReport:
    return EvaluationReport(
        experiment_type="within_project",
        dataset_summary={
            "snapshots": 4,
            "repository_count": 1,
            "repositories": ["owner/example"],
            "positive_labels": 2,
            "negative_labels": 2,
            "positive_rate": 0.5,
            "date_range": {"start": "2021-01-31", "end": "2021-04-30"},
        },
        selected_model="example",
        selection_metric="validation_pr_auc",
        metrics={"sentinel_version": "0.5.0", "models": {}},
        model_comparison=pd.DataFrame(
            [
                {
                    "model": "example",
                    "selected_model": True,
                    "threshold": 0.5,
                    "accuracy": 0.75,
                    "precision": 2 / 3,
                    "recall": 1.0,
                    "f1": 0.8,
                    "roc_auc": 1.0,
                    "pr_auc": 1.0,
                    "positive_prediction_rate": 0.75,
                }
            ]
        ),
        feature_importance=pd.DataFrame(
            [
                {
                    "model": "example",
                    "feature": "commit_count",
                    "importance": 0.8,
                    "effect": 0.8,
                    "direction": "not_applicable",
                    "method": "native_feature_importance",
                    "standard_deviation": None,
                    "rank": 1,
                    "fold_count": 1,
                }
            ]
        ),
        confusion_matrix=np.array([[1, 1], [0, 2]]),
        confusion_matrix_label="example test",
        key_findings=("The example model found both positive rows.",),
    )


def test_report_generation_writes_exact_v5_artifacts(tmp_path: Path) -> None:
    paths = save_evaluation_report(sample_report(), tmp_path / "report")

    assert set(paths) == set(REPORT_FILENAMES)
    assert {path.name for path in paths.values()} == set(REPORT_FILENAMES.values())
    assert all(path.exists() for path in paths.values())
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["experiment_type"] == "within_project"
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert pd.read_csv(paths["model_comparison"]).loc[0, "accuracy"] == 0.75
    assert pd.read_csv(paths["feature_importance"]).loc[0, "rank"] == 1
    assert paths["confusion_matrix"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    markdown = paths["summary"].read_text(encoding="utf-8")
    assert "# Sentinel V5 evaluation report" in markdown
    assert "## Key findings" in markdown


def test_within_project_cli_creates_complete_report(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "example_features.csv"
    make_feature_data("owner/example").to_csv(source, index=False)
    destination = tmp_path / "within-report"

    exit_code = main(
        [
            "within-project",
            str(source),
            "--output-dir",
            str(destination),
            "--random-state",
            "11",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "V5 report directory" in captured.out
    assert captured.err == ""
    assert {path.name for path in destination.iterdir()} == set(
        REPORT_FILENAMES.values()
    )


def test_cross_project_adapter_uses_same_report_contract(tmp_path: Path) -> None:
    sources: list[Path] = []
    for index, repository in enumerate(("owner/alpha", "owner/beta")):
        source = tmp_path / f"project-{index}.csv"
        make_feature_data(repository, offset=index).to_csv(source, index=False)
        sources.append(source)
    dataset = load_multi_repository_datasets(sources)

    report = evaluate_cross_project(dataset, random_state=13)

    assert report.experiment_type == "cross_project"
    assert set(report.model_comparison["model"]) == {
        "dummy",
        "logistic_regression",
        "random_forest",
    }
    assert "mean_accuracy" in report.model_comparison
    assert report.confusion_matrix.sum() == len(dataset.data)
    assert set(report.feature_importance["model"]) == set(
        report.model_comparison["model"]
    )
