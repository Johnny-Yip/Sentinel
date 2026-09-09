from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from sentinel.ml.cli import main
from sentinel.ml.data import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    InvalidDatasetError,
    validate_dataset,
)
from sentinel.ml.evaluation import calculate_metrics, select_f1_threshold
from sentinel.ml.models import MODEL_ORDER
from sentinel.ml.splitting import chronological_split
from sentinel.ml.training import save_artifacts, train_baselines


def make_feature_data(date_count: int = 20, files_per_date: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2020-01-31", periods=date_count, freq="ME")
    for date_index, snapshot_date in enumerate(dates):
        for file_index in range(files_per_date):
            commit_count = date_index + file_index + 1
            lines_added = commit_count * 7
            lines_deleted = commit_count * 3
            rows.append(
                {
                    "repository": "owner/repository",
                    "file_path": f"src/File{file_index}.java",
                    "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
                    "commit_count": commit_count,
                    "developer_count": file_index + 1,
                    "lines_added": lines_added,
                    "lines_deleted": lines_deleted,
                    "code_churn": lines_added + lines_deleted,
                    "file_age_days": date_index * 30 + file_index,
                    "days_since_last_change": (date_index * 3 + file_index) % 31,
                    "previous_bug_fixes": (date_index + file_index) % 4,
                    "defect_next_90_days": int(
                        file_index == 0 or (file_index == 1 and date_index % 5 == 0)
                    ),
                    # An identifier-like numeric extra must never become a feature.
                    "repository_id": 999,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def validated_data() -> pd.DataFrame:
    return validate_dataset(make_feature_data())


@pytest.fixture(scope="module")
def trained_result(validated_data):
    return train_baselines(chronological_split(validated_data), random_state=17)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="commit_count"), "missing required columns"),
        (
            lambda frame: frame.assign(commit_count="not-numeric"),
            "finite numeric values",
        ),
        (
            lambda frame: frame.assign(lines_added=np.nan),
            "missing values",
        ),
        (
            lambda frame: frame.assign(defect_next_90_days=2),
            "binary 0/1",
        ),
        (
            lambda frame: frame.assign(snapshot_date="not-a-date"),
            "invalid date",
        ),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "duplicate repository/file/snapshot",
        ),
    ],
)
def test_dataset_validation_rejects_malformed_input(mutate, message: str) -> None:
    with pytest.raises(InvalidDatasetError, match=message):
        validate_dataset(mutate(make_feature_data()))


def test_dataset_validation_normalizes_types(validated_data) -> None:
    assert pd.api.types.is_datetime64_ns_dtype(validated_data["snapshot_date"])
    assert validated_data[TARGET_COLUMN].dtype == np.dtype("int8")
    assert all(
        validated_data[column].dtype == np.dtype("float64")
        for column in MODEL_FEATURE_COLUMNS
    )


def test_chronological_split_uses_date_groups(validated_data) -> None:
    splits = chronological_split(validated_data)
    train_dates = set(splits.train["snapshot_date"])
    validation_dates = set(splits.validation["snapshot_date"])
    test_dates = set(splits.test["snapshot_date"])

    assert len(train_dates) == 14
    assert len(validation_dates) == 3
    assert len(test_dates) == 3
    assert max(train_dates) < min(validation_dates) < min(test_dates)
    assert train_dates.isdisjoint(validation_dates | test_dates)
    assert validation_dates.isdisjoint(test_dates)

    for snapshot_date, rows in validated_data.groupby("snapshot_date"):
        membership = sum(
            snapshot_date in dates
            for dates in (train_dates, validation_dates, test_dates)
        )
        assert membership == 1
        assert len(rows) == 4


def test_metadata_and_target_are_excluded_from_model_inputs(trained_result) -> None:
    assert trained_result.feature_columns == MODEL_FEATURE_COLUMNS
    for model_result in trained_result.models.values():
        preprocessor = model_result.pipeline.named_steps["preprocess"]
        assert list(preprocessor.feature_names_in_) == MODEL_FEATURE_COLUMNS
        assert "repository_id" not in preprocessor.feature_names_in_
        assert "file_path" not in preprocessor.feature_names_in_
        assert "snapshot_date" not in preprocessor.feature_names_in_
        assert TARGET_COLUMN not in preprocessor.feature_names_in_


def test_logistic_preprocessing_is_fitted_on_training_only(trained_result) -> None:
    scaler = trained_result.models["logistic_regression"].pipeline.named_steps[
        "preprocess"
    ].named_transformers_["numeric"]
    training_means = trained_result.splits.train[MODEL_FEATURE_COLUMNS].mean().to_numpy()
    all_means = pd.concat(
        [
            trained_result.splits.train,
            trained_result.splits.validation,
            trained_result.splits.test,
        ]
    )[MODEL_FEATURE_COLUMNS].mean().to_numpy()

    np.testing.assert_allclose(scaler.mean_, training_means)
    assert not np.allclose(scaler.mean_, all_means)


def test_all_baseline_models_train_and_evaluate(trained_result) -> None:
    assert tuple(trained_result.models) == MODEL_ORDER
    assert trained_result.best_model_name in MODEL_ORDER
    for model in trained_result.models.values():
        assert model.pipeline.classes_.tolist() == [0, 1]
        assert 0 <= model.selected_threshold <= 1
        for metrics in (
            model.validation_default,
            model.validation_selected,
            model.test_default,
            model.test_selected,
        ):
            assert set(metrics) == {
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "confusion_matrix",
                "positive_prediction_rate",
            }


def test_metric_calculation_includes_confusion_and_rare_class_metrics() -> None:
    metrics = calculate_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.8, 0.9]),
        threshold=0.5,
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["positive_prediction_rate"] == pytest.approx(0.75)


def test_metric_edge_case_without_positive_labels_is_clean() -> None:
    metrics = calculate_metrics(np.array([0, 0]), np.array([0.1, 0.2]))

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["accuracy"] == 1
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 0]]


def test_threshold_selection_maximizes_validation_f1() -> None:
    threshold = select_f1_threshold(
        np.array([0, 1, 1, 0]),
        np.array([0.1, 0.4, 0.35, 0.8]),
    )

    assert threshold == pytest.approx(0.35)


def test_test_labels_cannot_change_validation_threshold_selection(validated_data) -> None:
    splits = chronological_split(validated_data)
    first = train_baselines(splits, random_state=23)
    altered_test = splits.test.copy()
    altered_test[TARGET_COLUMN] = 1 - altered_test[TARGET_COLUMN]
    second = train_baselines(
        type(splits)(splits.train, splits.validation, altered_test), random_state=23
    )

    assert first.best_model_name == second.best_model_name
    for name in MODEL_ORDER:
        assert first.models[name].selected_threshold == pytest.approx(
            second.models[name].selected_threshold
        )
        assert first.models[name].feature_importance == (
            second.models[name].feature_importance
        )


def test_artifacts_include_pipeline_and_required_metadata(
    tmp_path: Path, trained_result
) -> None:
    model_path, metadata_path = save_artifacts(trained_result, tmp_path / "artifacts")

    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert pipeline.predict_proba(
        trained_result.splits.test[MODEL_FEATURE_COLUMNS].iloc[:2]
    ).shape == (2, 2)
    assert metadata["sentinel_version"] == "0.5.0"
    assert metadata["model_type"] == trained_result.best_model_name
    assert metadata["feature_columns"] == MODEL_FEATURE_COLUMNS
    assert metadata["selected_threshold"] == pytest.approx(
        trained_result.best_model.selected_threshold
    )
    assert set(metadata["date_ranges"]) == {"training", "validation", "test"}
    assert set(metadata["evaluation_metrics"]) == {"validation", "test"}
    assert metadata["leakage_audit"]["status"] == "passed"
    assert metadata["leakage_audit"]["future_derived_predictors"] == []


def test_training_is_reproducible(validated_data) -> None:
    splits = chronological_split(validated_data)
    first = train_baselines(splits, random_state=101)
    second = train_baselines(splits, random_state=101)

    assert first.best_model_name == second.best_model_name
    for name in MODEL_ORDER:
        assert first.models[name].selected_threshold == pytest.approx(
            second.models[name].selected_threshold
        )
        assert first.models[name].test_selected == second.models[name].test_selected
        assert (
            first.models[name].feature_importance
            == second.models[name].feature_importance
        )


def test_feature_importance_uses_native_and_fallback_methods(trained_result) -> None:
    expected_methods = {
        "dummy": "permutation_average_precision",
        "logistic_regression": "native_coefficient",
        "random_forest": "native_feature_importance",
    }
    for name, expected_method in expected_methods.items():
        ranked = trained_result.models[name].feature_importance["ranked_features"]
        assert len(ranked) == len(MODEL_FEATURE_COLUMNS)
        assert {record["method"] for record in ranked} == {expected_method}
        assert [record["rank"] for record in ranked] == list(
            range(1, len(MODEL_FEATURE_COLUMNS) + 1)
        )
        assert {record["feature"] for record in ranked} == set(
            MODEL_FEATURE_COLUMNS
        )


def test_cli_reports_malformed_input_without_artifacts(tmp_path: Path, capsys) -> None:
    source = tmp_path / "malformed.csv"
    make_feature_data().drop(columns="snapshot_date").to_csv(source, index=False)
    output_dir = tmp_path / "artifacts"

    exit_code = main([str(source), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err
    assert "snapshot_date" in captured.err
    assert not output_dir.exists()
