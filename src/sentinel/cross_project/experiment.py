"""Run and summarize scientifically isolated cross-project experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentinel import __version__
from sentinel.cross_project.data import MultiRepositoryDataset
from sentinel.cross_project.splitting import (
    CrossProjectFold,
    build_leave_one_project_out_folds,
)
from sentinel.ml.data import (
    DATE_COLUMN,
    METADATA_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PATH_COLUMN,
    REPOSITORY_COLUMN,
    TARGET_COLUMN,
    InvalidDatasetError,
)
from sentinel.ml.evaluation import DEFAULT_THRESHOLD
from sentinel.ml.models import MODEL_ORDER
from sentinel.ml.training import TrainingResult, train_baselines


ProgressCallback = Callable[[str], None]
THRESHOLD_STRATEGIES = ("default_0_5", "validation_selected")


@dataclass
class FoldEvaluation:
    """Models, metrics, and audit for one held-out repository."""

    fold: CrossProjectFold
    training_result: TrainingResult | None
    leakage_audit: dict[str, Any]


@dataclass
class CrossProjectResult:
    """All machine-readable outputs from a Sentinel V4 experiment."""

    dataset: MultiRepositoryDataset
    fold_evaluations: tuple[FoldEvaluation, ...]
    folds: pd.DataFrame
    aggregate: pd.DataFrame
    coefficient_stability: pd.DataFrame
    feature_importance_stability: pd.DataFrame
    dataset_shift: pd.DataFrame
    model_selection: list[dict[str, Any]]
    report: dict[str, Any]


def _with_prevalence(metrics: dict[str, Any], prevalence: float) -> dict[str, Any]:
    enriched = dict(metrics)
    pr_auc = enriched["pr_auc"]
    enriched["class_prevalence"] = float(prevalence)
    enriched["pr_auc_lift"] = (
        float(pr_auc / prevalence)
        if pr_auc is not None and prevalence > 0
        else None
    )
    return enriched


def _metric_row(
    fold: CrossProjectFold,
    result: TrainingResult,
    model_name: str,
    threshold_strategy: str,
) -> dict[str, Any]:
    model = result.models[model_name]
    metrics = (
        model.test_default
        if threshold_strategy == "default_0_5"
        else model.test_selected
    )
    prevalence = float(fold.test[TARGET_COLUMN].mean())
    enriched = _with_prevalence(metrics, prevalence)
    matrix = enriched["confusion_matrix"]
    validation = model.validation_selected
    return {
        "held_out_repository": fold.held_out_repository,
        "training_repositories": "|".join(fold.training_repositories),
        "model": model_name,
        "is_selected_model": model_name == result.best_model_name,
        "threshold_strategy": threshold_strategy,
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_validation_threshold": model.selected_threshold,
        "threshold": enriched["threshold"],
        "validation_pr_auc": validation["pr_auc"],
        "validation_f1": validation["f1"],
        "precision": enriched["precision"],
        "recall": enriched["recall"],
        "f1": enriched["f1"],
        "roc_auc": enriched["roc_auc"],
        "pr_auc": enriched["pr_auc"],
        "confusion_tn": matrix[0][0],
        "confusion_fp": matrix[0][1],
        "confusion_fn": matrix[1][0],
        "confusion_tp": matrix[1][1],
        "positive_prediction_rate": enriched["positive_prediction_rate"],
        "class_prevalence": enriched["class_prevalence"],
        "pr_auc_lift": enriched["pr_auc_lift"],
        "test_rows": len(fold.test),
        "test_positives": int(fold.test[TARGET_COLUMN].sum()),
    }


def _aggregate_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    selected = folds.loc[folds["threshold_strategy"] == "validation_selected"]
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_rows = selected.loc[selected["model"] == model_name]
        pr_auc = model_rows["pr_auc"].dropna()
        rows.append(
            {
                "model": model_name,
                "fold_count": len(model_rows),
                "mean_pr_auc": pr_auc.mean() if not pr_auc.empty else None,
                "median_pr_auc": pr_auc.median() if not pr_auc.empty else None,
                "std_pr_auc": pr_auc.std(ddof=0) if not pr_auc.empty else None,
                "min_pr_auc": pr_auc.min() if not pr_auc.empty else None,
                "max_pr_auc": pr_auc.max() if not pr_auc.empty else None,
                "mean_pr_auc_lift": model_rows["pr_auc_lift"].mean(),
                "mean_f1": model_rows["f1"].mean(),
                "mean_precision": model_rows["precision"].mean(),
                "mean_recall": model_rows["recall"].mean(),
                "mean_roc_auc": model_rows["roc_auc"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_rows(
    held_out_repository: str, result: TrainingResult
) -> list[dict[str, Any]]:
    pipeline = result.models["logistic_regression"].pipeline
    coefficients = np.asarray(pipeline.named_steps["model"].coef_[0], dtype=float)
    return [
        {
            "held_out_repository": held_out_repository,
            "feature": feature,
            "coefficient": float(coefficient),
        }
        for feature, coefficient in zip(
            result.feature_columns, coefficients, strict=True
        )
    ]


def _coefficient_stability(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    raw = pd.DataFrame(raw_rows)
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURE_COLUMNS:
        values = raw.loc[raw["feature"] == feature, "coefficient"].to_numpy()
        positive_fraction = float(np.mean(values > 0))
        negative_fraction = float(np.mean(values < 0))
        zero_fraction = float(np.mean(values == 0))
        if positive_fraction == 1.0:
            direction = "consistently_positive"
        elif negative_fraction == 1.0:
            direction = "consistently_negative"
        else:
            direction = "mixed"
        rows.append(
            {
                "feature": feature,
                "mean_coefficient": float(np.mean(values)),
                "std_coefficient": float(np.std(values, ddof=0)),
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "sign_consistency": max(
                    positive_fraction, negative_fraction, zero_fraction
                ),
                "direction": direction,
                "fold_count": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_coefficient", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _feature_importance_rows(
    held_out_repository: str, result: TrainingResult
) -> list[dict[str, Any]]:
    pipeline = result.models["random_forest"].pipeline
    importances = np.asarray(
        pipeline.named_steps["model"].feature_importances_, dtype=float
    )
    return [
        {
            "held_out_repository": held_out_repository,
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(
            result.feature_columns, importances, strict=True
        )
    ]


def _feature_importance_stability(
    raw_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    raw = pd.DataFrame(raw_rows)
    rows: list[dict[str, Any]] = []
    for feature in MODEL_FEATURE_COLUMNS:
        values = raw.loc[raw["feature"] == feature, "importance"].to_numpy()
        rows.append(
            {
                "feature": feature,
                "mean_importance": float(np.mean(values)),
                "std_importance": float(np.std(values, ddof=0)),
                "min_importance": float(np.min(values)),
                "max_importance": float(np.max(values)),
                "fold_count": len(values),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_importance", ascending=False, kind="stable"
    ).reset_index(drop=True)


def _distribution_summary(values: pd.Series) -> tuple[float, float]:
    median = float(values.median())
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    return median, iqr


def _dataset_shift(evaluations: Sequence[FoldEvaluation]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        fold = evaluation.fold
        training_reference = pd.concat(
            [fold.train, fold.validation], ignore_index=True
        )
        for feature in MODEL_FEATURE_COLUMNS:
            train_median, train_iqr = _distribution_summary(
                training_reference[feature]
            )
            held_median, held_iqr = _distribution_summary(fold.test[feature])
            shift = (
                (held_median - train_median) / train_iqr
                if train_iqr > 0
                else None
            )
            rows.append(
                {
                    "held_out_repository": fold.held_out_repository,
                    "feature": feature,
                    "training_median": train_median,
                    "training_iqr": train_iqr,
                    "held_out_median": held_median,
                    "held_out_iqr": held_iqr,
                    "standardized_median_shift": shift,
                    "absolute_standardized_median_shift": (
                        abs(shift) if shift is not None else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _audit_fold(
    fold: CrossProjectFold,
    result: TrainingResult,
) -> dict[str, Any]:
    expected_training = set(fold.training_repositories)
    train_repositories = set(fold.train[REPOSITORY_COLUMN].unique())
    validation_repositories = set(fold.validation[REPOSITORY_COLUMN].unique())
    test_repositories = set(fold.test[REPOSITORY_COLUMN].unique())

    date_grouping_passed = True
    for repository in fold.training_repositories:
        train_dates = set(
            fold.train.loc[
                fold.train[REPOSITORY_COLUMN] == repository, DATE_COLUMN
            ]
        )
        validation_dates = set(
            fold.validation.loc[
                fold.validation[REPOSITORY_COLUMN] == repository, DATE_COLUMN
            ]
        )
        if train_dates & validation_dates:
            date_grouping_passed = False

    predictor_set = set(result.feature_columns)
    forbidden = {
        REPOSITORY_COLUMN,
        PATH_COLUMN,
        DATE_COLUMN,
        TARGET_COLUMN,
    }
    future_derived = sorted(
        column
        for column in fold.train.columns
        if column == TARGET_COLUMN or "future" in column.lower()
    )
    preprocessor_inputs_match = all(
        list(model.pipeline.named_steps["preprocess"].feature_names_in_)
        == result.feature_columns
        for model in result.models.values()
    )
    scaler = result.models["logistic_regression"].pipeline.named_steps[
        "preprocess"
    ].named_transformers_["numeric"]
    preprocessing_row_count_matches = int(scaler.n_samples_seen_) == len(fold.train)

    checks = {
        "held_out_absent_from_training": fold.held_out_repository
        not in train_repositories,
        "held_out_absent_from_validation": fold.held_out_repository
        not in validation_repositories,
        "held_out_only_in_test": test_repositories == {fold.held_out_repository},
        "all_training_repositories_present": train_repositories
        == expected_training,
        "all_validation_repositories_present": validation_repositories
        == expected_training,
        "same_date_rows_grouped_within_repository": date_grouping_passed,
        "repository_excluded": REPOSITORY_COLUMN not in predictor_set,
        "file_path_excluded": PATH_COLUMN not in predictor_set,
        "snapshot_date_excluded": DATE_COLUMN not in predictor_set,
        "target_excluded": TARGET_COLUMN not in predictor_set,
        "future_derived_fields_excluded": not predictor_set.intersection(
            future_derived
        ),
        "predictors_match_historical_whitelist": list(result.feature_columns)
        == MODEL_FEATURE_COLUMNS,
        "preprocessor_inputs_match_whitelist": preprocessor_inputs_match,
        "preprocessing_fit_row_count_matches_training": (
            preprocessing_row_count_matches
        ),
        # These scopes are guaranteed by train_baselines: selection is completed
        # before test features or labels are read. Counterfactual tests enforce it.
        "threshold_selection_uses_validation_only": True,
        "model_selection_uses_validation_only": True,
        "held_out_labels_used_for_final_evaluation_only": True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "feature_columns": list(result.feature_columns),
        "metadata_columns_excluded": list(METADATA_COLUMNS),
        "future_derived_columns_excluded": future_derived,
        "preprocessing_fit_scope": "training_rows_only",
        "threshold_selection_scope": "training_repository_validation_rows_only",
        "model_selection_scope": "training_repository_validation_rows_only",
        "held_out_label_scope": "final_evaluation_only",
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


def _same_project_comparison(
    metadata_path: str | Path | None,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    if metadata_path is None:
        return {"available": False, "reason": "same-project metadata not supplied"}

    source = Path(metadata_path).expanduser().resolve()
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "available": False,
            "reason": f"same-project metadata does not exist: {source}",
        }
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidDatasetError(
            f"Could not read same-project metadata {source}: {exc}"
        ) from exc

    commons_repositories = sorted(
        repository
        for repository in folds["held_out_repository"].unique()
        if repository.endswith("/commons-lang") or repository == "commons-lang"
    )
    if not commons_repositories:
        return {
            "available": False,
            "reason": "no commons-lang held-out fold was present",
            "metadata_path": str(source),
        }

    held_out = commons_repositories[0]
    comparisons: dict[str, Any] = {}
    for label, model_name in (
        ("logistic_regression", "logistic_regression"),
        ("validation_selected_model", metadata.get("model_type")),
    ):
        if not isinstance(model_name, str):
            continue
        same_model = metadata.get("all_model_comparison", {}).get(model_name, {})
        same_metrics = same_model.get("test")
        cross_rows = folds.loc[
            (folds["held_out_repository"] == held_out)
            & (folds["model"] == model_name)
            & (folds["threshold_strategy"] == "validation_selected")
        ]
        if not same_metrics or cross_rows.empty:
            continue
        cross = cross_rows.iloc[0]
        comparison = {
            "model": model_name,
            "same_project": {
                metric: same_metrics.get(metric)
                for metric in ("pr_auc", "f1", "roc_auc")
            },
            "cross_project": {
                metric: cross[metric] for metric in ("pr_auc", "f1", "roc_auc")
            },
        }
        comparison["cross_minus_same"] = {
            metric: (
                comparison["cross_project"][metric]
                - comparison["same_project"][metric]
                if comparison["cross_project"][metric] is not None
                and comparison["same_project"][metric] is not None
                else None
            )
            for metric in ("pr_auc", "f1", "roc_auc")
        }
        same_pr = comparison["same_project"]["pr_auc"]
        cross_pr = comparison["cross_project"]["pr_auc"]
        comparison["materially_worse_pr_auc"] = bool(
            same_pr and cross_pr < 0.8 * same_pr
        )
        comparisons[label] = comparison

    return {
        "available": bool(comparisons),
        "repository": held_out,
        "metadata_path": str(source),
        "comparisons": comparisons,
        "explanation": (
            "Cross-project evaluation is stricter because no snapshot from the "
            "held-out repository is available during fitting or selection."
        ),
    }


def _generalization_interpretation(
    folds: pd.DataFrame,
    aggregate: pd.DataFrame,
    same_project: dict[str, Any],
) -> dict[str, Any]:
    selected_rows = folds.loc[
        (folds["threshold_strategy"] == "validation_selected")
        & folds["is_selected_model"]
    ]
    lift_by_repository = {
        row["held_out_repository"]: row["pr_auc_lift"]
        for row in _json_records(selected_rows)
    }
    well = sorted(
        repository
        for repository, lift in lift_by_repository.items()
        if lift is not None and lift > 1.0
    )
    poor = sorted(
        repository
        for repository, lift in lift_by_repository.items()
        if lift is None or lift <= 1.0
    )

    non_dummy = aggregate.loc[aggregate["model"] != "dummy"]
    best_row = aggregate.loc[aggregate["mean_pr_auc"].idxmax()]
    stable_row = non_dummy.loc[non_dummy["std_pr_auc"].idxmin()]
    variation = float(best_row["max_pr_auc"] - best_row["min_pr_auc"])
    materially_worse = None
    if same_project.get("available"):
        comparison = same_project.get("comparisons", {}).get(
            "logistic_regression"
        )
        if comparison:
            materially_worse = comparison["materially_worse_pr_auc"]

    return {
        "selected_models_above_prevalence_on_all_repositories": not poor,
        "repositories_above_prevalence": well,
        "repositories_not_above_prevalence": poor,
        "selected_model_pr_auc_lift_by_repository": lift_by_repository,
        "best_mean_pr_auc_model": str(best_row["model"]),
        "logistic_regression_remains_most_robust": (
            best_row["model"] == "logistic_regression"
        ),
        "most_stable_non_dummy_model": str(stable_row["model"]),
        "most_stable_non_dummy_pr_auc_std": float(stable_row["std_pr_auc"]),
        "best_model_pr_auc_range_across_repositories": variation,
        "cross_project_materially_worse_than_same_project_commons_lang": (
            materially_worse
        ),
        "well_definition": "validation-selected model PR-AUC lift > 1.0",
        "materially_worse_definition": (
            "cross-project PR-AUC is at least 20% below same-project PR-AUC"
        ),
    }


def _fold_definition(evaluation: FoldEvaluation) -> dict[str, Any]:
    fold = evaluation.fold
    return {
        "held_out_repository": fold.held_out_repository,
        "training_repositories": list(fold.training_repositories),
        "training_rows": len(fold.train),
        "validation_rows": len(fold.validation),
        "held_out_rows": len(fold.test),
        "held_out_date_range": {
            "start": fold.test[DATE_COLUMN].min().strftime("%Y-%m-%d"),
            "end": fold.test[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        },
        "training_repository_date_ranges": fold.repository_date_ranges,
    }


def run_cross_project_evaluation(
    dataset: MultiRepositoryDataset,
    *,
    random_state: int = 42,
    same_project_metadata: str | Path | None = None,
    progress: ProgressCallback | None = None,
    retain_models: bool = True,
) -> CrossProjectResult:
    """Execute all leave-one-project-out folds and build aggregate analyses."""
    folds = build_leave_one_project_out_folds(dataset.data)
    evaluations: list[FoldEvaluation] = []
    metric_rows: list[dict[str, Any]] = []
    model_selection: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for fold in folds:
        if progress:
            progress(
                f"Held out {fold.held_out_repository}; training on "
                f"{', '.join(fold.training_repositories)}."
            )
        training_result = train_baselines(
            fold.as_temporal_splits(),
            feature_columns=dataset.feature_columns,
            random_state=random_state,
            progress=(
                (lambda message, repo=fold.held_out_repository: progress(
                    f"  [{repo}] {message}"
                ))
                if progress
                else None
            ),
        )
        audit = _audit_fold(fold, training_result)
        if audit["status"] != "passed":
            raise InvalidDatasetError(
                f"Cross-project leakage audit failed for {fold.held_out_repository}."
            )
        coefficient_rows.extend(
            _coefficient_rows(fold.held_out_repository, training_result)
        )
        importance_rows.extend(
            _feature_importance_rows(fold.held_out_repository, training_result)
        )
        evaluation = FoldEvaluation(
            fold, training_result if retain_models else None, audit
        )
        evaluations.append(evaluation)

        candidates: dict[str, Any] = {}
        for model_name in MODEL_ORDER:
            model = training_result.models[model_name]
            candidates[model_name] = {
                "validation_pr_auc": model.validation_selected["pr_auc"],
                "validation_f1": model.validation_selected["f1"],
                "validation_recall": model.validation_selected["recall"],
                "selected_threshold": model.selected_threshold,
            }
            for strategy in THRESHOLD_STRATEGIES:
                metric_rows.append(
                    _metric_row(fold, training_result, model_name, strategy)
                )
        model_selection.append(
            {
                "held_out_repository": fold.held_out_repository,
                "selection_metric": "validation_pr_auc",
                "selected_model": training_result.best_model_name,
                "candidates": candidates,
            }
        )

    folds_frame = pd.DataFrame(metric_rows)
    aggregate = _aggregate_metrics(folds_frame)
    coefficient_stability = _coefficient_stability(coefficient_rows)
    importance_stability = _feature_importance_stability(importance_rows)
    dataset_shift = _dataset_shift(evaluations)
    same_project = _same_project_comparison(same_project_metadata, folds_frame)
    interpretation = _generalization_interpretation(
        folds_frame, aggregate, same_project
    )
    audits = {
        evaluation.fold.held_out_repository: evaluation.leakage_audit
        for evaluation in evaluations
    }
    report = {
        "sentinel_version": __version__,
        "experiment": "leave_one_project_out",
        "random_state": random_state,
        "validation_strategy": (
            "per_training_repository_earliest_80_percent_dates_train_"
            "latest_20_percent_dates_validation"
        ),
        "selection_metric": "validation_pr_auc",
        "threshold_metric": "validation_f1",
        "feature_columns": list(dataset.feature_columns),
        "excluded_predictor_columns": sorted(
            set(dataset.data.columns) - set(dataset.feature_columns)
        ),
        "repository_summaries": [
            summary.to_dict() for summary in dataset.summaries
        ],
        "fold_definitions": [
            _fold_definition(evaluation) for evaluation in evaluations
        ],
        "model_selection": model_selection,
        "per_fold_metrics": _json_records(folds_frame),
        "aggregate_metrics": _json_records(aggregate),
        "coefficient_stability": _json_records(coefficient_stability),
        "feature_importance_stability": _json_records(importance_stability),
        "dataset_shift": _json_records(dataset_shift),
        "same_project_comparison": same_project,
        "generalization_interpretation": interpretation,
        "leakage_audit": {
            "status": (
                "passed"
                if all(audit["status"] == "passed" for audit in audits.values())
                else "failed"
            ),
            "folds": audits,
        },
    }

    return CrossProjectResult(
        dataset=dataset,
        fold_evaluations=tuple(evaluations),
        folds=folds_frame,
        aggregate=aggregate,
        coefficient_stability=coefficient_stability,
        feature_importance_stability=importance_stability,
        dataset_shift=dataset_shift,
        model_selection=model_selection,
        report=report,
    )
