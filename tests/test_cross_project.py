from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentinel.cross_project.cli import main
from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.cross_project.experiment import run_cross_project_evaluation
from sentinel.cross_project.reporting import save_cross_project_artifacts
from sentinel.cross_project.splitting import build_leave_one_project_out_folds
from sentinel.ml.data import MODEL_FEATURE_COLUMNS, TARGET_COLUMN, InvalidDatasetError
from sentinel.ml.models import MODEL_ORDER


def make_repository_data(
    repository: str, *, date_count: int = 10, offset: int = 0
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date_index, snapshot_date in enumerate(
        pd.date_range("2020-01-31", periods=date_count, freq="ME")
    ):
        for file_index in range(4):
            commit_count = date_index + file_index + offset + 1
            lines_added = commit_count * 5
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
                    "days_since_last_change": (date_index * 2 + file_index) % 17,
                    "previous_bug_fixes": (date_index + file_index + offset) % 4,
                    "defect_next_90_days": int(
                        file_index == 0
                        or (file_index == 1 and (date_index + offset) % 4 == 0)
                    ),
                    "repository_id": offset + 100,
                    "future_outcome_debug": date_index,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def feature_paths(tmp_path: Path) -> list[Path]:
    paths: list[Path] = []
    for index, repository in enumerate(("owner/alpha", "owner/beta", "owner/gamma")):
        path = tmp_path / f"{repository.rsplit('/', 1)[1]}_features.csv"
        make_repository_data(repository, offset=index).to_csv(path, index=False)
        paths.append(path)
    return paths


@pytest.fixture()
def multi_dataset(feature_paths):
    return load_multi_repository_datasets(feature_paths)


@pytest.fixture()
def cross_project_result(multi_dataset):
    return run_cross_project_evaluation(multi_dataset, random_state=19)


def test_repository_metadata_and_summaries_are_preserved(multi_dataset) -> None:
    assert set(multi_dataset.data["repository"]) == {
        "owner/alpha",
        "owner/beta",
        "owner/gamma",
    }
    assert multi_dataset.feature_columns == tuple(MODEL_FEATURE_COLUMNS)
    assert [summary.repository for summary in multi_dataset.summaries] == [
        "owner/alpha",
        "owner/beta",
        "owner/gamma",
    ]
    for summary in multi_dataset.summaries:
        assert summary.snapshot_count == 40
        assert summary.unique_java_files == 4
        assert summary.positive_labels + summary.negative_labels == 40
        assert summary.positive_rate == pytest.approx(
            summary.positive_labels / summary.snapshot_count
        )
        assert summary.start_date == "2020-01-31"
        assert summary.end_date == "2020-10-31"
        assert Path(summary.feature_csv_path).exists()


def test_multi_dataset_schema_validation_rejects_missing_predictor(
    feature_paths, tmp_path: Path
) -> None:
    malformed = tmp_path / "malformed.csv"
    make_repository_data("owner/delta").drop(columns="commit_count").to_csv(
        malformed, index=False
    )

    with pytest.raises(InvalidDatasetError, match="missing required columns"):
        load_multi_repository_datasets([feature_paths[0], malformed])


def test_duplicate_snapshot_across_inputs_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    make_repository_data("owner/alpha").to_csv(first, index=False)
    make_repository_data("owner/alpha").to_csv(second, index=False)

    with pytest.raises(InvalidDatasetError, match="more than one input CSV"):
        load_multi_repository_datasets([first, second])


def test_duplicate_snapshot_within_dataset_is_rejected(tmp_path: Path) -> None:
    malformed = tmp_path / "duplicate.csv"
    other = tmp_path / "other.csv"
    frame = make_repository_data("owner/alpha")
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(
        malformed, index=False
    )
    make_repository_data("owner/beta").to_csv(other, index=False)

    with pytest.raises(
        InvalidDatasetError, match="duplicate repository/file/snapshot"
    ):
        load_multi_repository_datasets([malformed, other])


def test_single_repository_input_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "only.csv"
    make_repository_data("owner/only").to_csv(source, index=False)

    with pytest.raises(InvalidDatasetError, match="at least two distinct repositories"):
        load_multi_repository_datasets([source])


def test_leave_one_project_out_fold_construction_and_date_grouping(
    multi_dataset,
) -> None:
    folds = build_leave_one_project_out_folds(multi_dataset.data)
    assert len(folds) == 3

    for fold in folds:
        assert set(fold.train["repository"]) == set(fold.training_repositories)
        assert set(fold.validation["repository"]) == set(fold.training_repositories)
        assert set(fold.test["repository"]) == {fold.held_out_repository}
        assert fold.held_out_repository not in set(fold.train["repository"])
        assert fold.held_out_repository not in set(fold.validation["repository"])

        for repository in fold.training_repositories:
            repository_all = multi_dataset.data.loc[
                multi_dataset.data["repository"] == repository
            ]
            train = fold.train.loc[fold.train["repository"] == repository]
            validation = fold.validation.loc[
                fold.validation["repository"] == repository
            ]
            assert train["snapshot_date"].nunique() == 8
            assert validation["snapshot_date"].nunique() == 2
            assert set(train["snapshot_date"]).isdisjoint(validation["snapshot_date"])
            assert max(train["snapshot_date"]) < min(validation["snapshot_date"])
            for date, rows in repository_all.groupby("snapshot_date"):
                expected = len(rows)
                actual = int((train["snapshot_date"] == date).sum()) + int(
                    (validation["snapshot_date"] == date).sum()
                )
                assert actual == expected


def test_repository_and_all_metadata_are_excluded_from_predictors(
    cross_project_result,
) -> None:
    forbidden = {
        "repository",
        "file_path",
        "snapshot_date",
        TARGET_COLUMN,
        "repository_id",
        "future_outcome_debug",
    }
    for evaluation in cross_project_result.fold_evaluations:
        assert set(evaluation.training_result.feature_columns).isdisjoint(forbidden)
        for model in evaluation.training_result.models.values():
            actual = list(
                model.pipeline.named_steps["preprocess"].feature_names_in_
            )
            assert actual == MODEL_FEATURE_COLUMNS
            assert set(actual).isdisjoint(forbidden)


def test_preprocessing_is_fitted_on_fold_training_rows_only(
    cross_project_result,
) -> None:
    for evaluation in cross_project_result.fold_evaluations:
        scaler = evaluation.training_result.models[
            "logistic_regression"
        ].pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        train_means = evaluation.fold.train[MODEL_FEATURE_COLUMNS].mean().to_numpy()
        all_means = pd.concat(
            [evaluation.fold.train, evaluation.fold.validation, evaluation.fold.test]
        )[MODEL_FEATURE_COLUMNS].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, train_means)
        assert int(scaler.n_samples_seen_) == len(evaluation.fold.train)
        assert not np.allclose(scaler.mean_, all_means)


def test_held_out_labels_cannot_change_threshold_or_model_selection(
    feature_paths,
) -> None:
    original_dataset = load_multi_repository_datasets(feature_paths)
    original = run_cross_project_evaluation(original_dataset, random_state=29)

    changed_path = feature_paths[0]
    changed = pd.read_csv(changed_path)
    changed[TARGET_COLUMN] = 1 - changed[TARGET_COLUMN]
    changed.to_csv(changed_path, index=False)
    altered_dataset = load_multi_repository_datasets(feature_paths)
    altered = run_cross_project_evaluation(altered_dataset, random_state=29)

    original_fold = next(
        evaluation
        for evaluation in original.fold_evaluations
        if evaluation.fold.held_out_repository == "owner/alpha"
    )
    altered_fold = next(
        evaluation
        for evaluation in altered.fold_evaluations
        if evaluation.fold.held_out_repository == "owner/alpha"
    )
    assert original_fold.training_result.best_model_name == (
        altered_fold.training_result.best_model_name
    )
    for model_name in MODEL_ORDER:
        assert original_fold.training_result.models[
            model_name
        ].selected_threshold == pytest.approx(
            altered_fold.training_result.models[model_name].selected_threshold
        )
        assert original_fold.training_result.models[
            model_name
        ].feature_importance == altered_fold.training_result.models[
            model_name
        ].feature_importance


def test_cross_project_run_is_reproducible(multi_dataset) -> None:
    first = run_cross_project_evaluation(multi_dataset, random_state=41)
    second = run_cross_project_evaluation(multi_dataset, random_state=41)

    pd.testing.assert_frame_equal(first.folds, second.folds)
    pd.testing.assert_frame_equal(first.aggregate, second.aggregate)
    pd.testing.assert_frame_equal(
        first.coefficient_stability, second.coefficient_stability
    )
    pd.testing.assert_frame_equal(
        first.feature_importance_stability,
        second.feature_importance_stability,
    )
    assert first.model_selection == second.model_selection
    pd.testing.assert_frame_equal(
        first.model_feature_importance, second.model_feature_importance
    )


def test_aggregate_metrics_and_pr_auc_lift_are_calculated(
    cross_project_result,
) -> None:
    selected = cross_project_result.folds.loc[
        cross_project_result.folds["threshold_strategy"] == "validation_selected"
    ]
    for row in selected.itertuples(index=False):
        assert row.pr_auc_lift == pytest.approx(row.pr_auc / row.class_prevalence)

    logistic_rows = selected.loc[selected["model"] == "logistic_regression"]
    aggregate = cross_project_result.aggregate.set_index("model").loc[
        "logistic_regression"
    ]
    assert aggregate["mean_pr_auc"] == pytest.approx(logistic_rows["pr_auc"].mean())
    assert aggregate["median_pr_auc"] == pytest.approx(
        logistic_rows["pr_auc"].median()
    )
    assert aggregate["std_pr_auc"] == pytest.approx(
        logistic_rows["pr_auc"].std(ddof=0)
    )
    assert aggregate["mean_pr_auc_lift"] == pytest.approx(
        logistic_rows["pr_auc_lift"].mean()
    )
    assert aggregate["mean_accuracy"] == pytest.approx(
        logistic_rows["accuracy"].mean()
    )


def test_coefficient_and_feature_importance_stability(cross_project_result) -> None:
    coefficients = cross_project_result.coefficient_stability
    importances = cross_project_result.feature_importance_stability
    assert set(coefficients["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert set(importances["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert set(coefficients["direction"]) <= {
        "consistently_positive",
        "consistently_negative",
        "mixed",
    }
    assert coefficients["sign_consistency"].between(0, 1).all()
    assert coefficients["fold_count"].eq(3).all()
    assert importances["fold_count"].eq(3).all()
    assert importances["mean_importance"].sum() == pytest.approx(1.0)
    assert len(cross_project_result.model_feature_importance) == (
        3 * len(MODEL_ORDER) * len(MODEL_FEATURE_COLUMNS)
    )

    feature = coefficients.iloc[0]["feature"]
    actual = []
    for evaluation in cross_project_result.fold_evaluations:
        model = evaluation.training_result.models["logistic_regression"].pipeline
        index = MODEL_FEATURE_COLUMNS.index(feature)
        actual.append(model.named_steps["model"].coef_[0][index])
    reported = coefficients.set_index("feature").loc[feature]
    assert reported["mean_coefficient"] == pytest.approx(np.mean(actual))
    assert reported["std_coefficient"] == pytest.approx(np.std(actual, ddof=0))


def test_dataset_shift_contains_robust_statistics(cross_project_result) -> None:
    shift = cross_project_result.dataset_shift
    assert len(shift) == 3 * len(MODEL_FEATURE_COLUMNS)
    assert set(shift["feature"]) == set(MODEL_FEATURE_COLUMNS)
    assert {
        "training_median",
        "training_iqr",
        "held_out_median",
        "held_out_iqr",
        "standardized_median_shift",
    } <= set(shift.columns)
    assert (shift["training_iqr"] >= 0).all()
    assert (shift["held_out_iqr"] >= 0).all()


def test_cross_project_leakage_audit_passes(cross_project_result) -> None:
    audit = cross_project_result.report["leakage_audit"]
    assert audit["status"] == "passed"
    for fold_audit in audit["folds"].values():
        assert fold_audit["status"] == "passed"
        assert all(fold_audit["checks"].values())
        assert fold_audit["held_out_label_scope"] == "final_evaluation_only"


def test_all_cross_project_artifacts_are_created(
    tmp_path: Path, cross_project_result
) -> None:
    paths = save_cross_project_artifacts(
        cross_project_result, tmp_path / "cross-project"
    )
    assert set(paths) == {
        "folds",
        "aggregate",
        "model_selection",
        "selected_thresholds",
        "coefficient_stability",
        "feature_importance_stability",
        "dataset_shift",
        "report_json",
        "report_markdown",
    }
    assert all(path.exists() for path in paths.values())
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert report["sentinel_version"] == "0.5.0"
    assert report["leakage_audit"]["status"] == "passed"
    assert len(pd.read_csv(paths["folds"])) == 3 * 3 * 2
    assert len(pd.read_csv(paths["aggregate"])) == 3
    assert len(pd.read_csv(paths["selected_thresholds"])) == 3 * 3


def test_cli_creates_artifacts_and_reports_audit(
    feature_paths, tmp_path: Path, capsys
) -> None:
    output_dir = tmp_path / "cli-artifacts"
    exit_code = main(
        [
            *(str(path) for path in feature_paths),
            "--output-dir",
            str(output_dir),
            "--random-state",
            "7",
            "--same-project-metadata",
            str(tmp_path / "missing.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Cross-project leakage audit: PASS" in captured.out
    assert "Same-project comparison unavailable" in captured.out
    assert (output_dir / "report.json").exists()
    assert captured.err == ""
