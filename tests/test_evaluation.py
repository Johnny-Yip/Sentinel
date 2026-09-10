from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.cross_project.data import load_multi_repository_datasets
from sentinel.evaluation.cli import main
from sentinel.evaluation.explain import (
    EXPLANATION_COLUMNS,
    add_top_explanations_to_ranking,
    build_explanation_summary,
    explain_predictions,
)
from sentinel.evaluation.experiment import (
    EvaluationReport,
    evaluate_cross_project,
)
from sentinel.evaluation.insights import (
    ACTIONABLE_INSIGHTS_SCHEMA_VERSION,
    RANKING_INSIGHT_COLUMNS,
    add_actionable_insights_to_ranking,
    build_actionable_insights,
)
from sentinel.evaluation.reporting import REPORT_FILENAMES, save_evaluation_report
from sentinel.evaluation.project_intelligence import (
    DEVELOPER_PRIORITY_COLUMNS,
    PROJECT_INTELLIGENCE_SCHEMA_VERSION,
)
from sentinel.evaluation.risk import (
    RISK_RANKING_COLUMNS,
    build_risk_ranking,
    build_risk_summary,
    calculate_risk_scores,
)


class ProbabilityClassifier:
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        column = "score" if "score" in features else features.columns[0]
        positive = np.asarray(features[column], dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def decision_function(self, features):  # pragma: no cover - must not be used
        raise AssertionError("predict_proba must be preferred")


class DecisionClassifier:
    classes_ = np.array([0, 1])

    def decision_function(self, features):
        return np.asarray(features["margin"], dtype=float)


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
    scores = np.array([0.1, 0.9, 0.4, 0.8])
    samples = pd.DataFrame(
        {
            "repository": ["owner/example"] * 4,
            "file_path": [f"src/File{index}.java" for index in range(4)],
            "snapshot_date": pd.date_range("2021-01-31", periods=4, freq="ME"),
            "defect_next_90_days": [0, 1, 0, 1],
            "score": scores,
            "commit_count": scores,
        }
    )
    risk_ranking = build_risk_ranking(
        samples,
        scores,
        model="example",
        evaluation_mode="within_project_temporal_test",
        score_method="predict_proba",
        decision_threshold=0.5,
    )
    prediction_explanations = explain_predictions(
        ProbabilityClassifier(),
        samples,
        ["commit_count"],
        scores,
        model="example",
        evaluation_mode="within_project_temporal_test",
        decision_threshold=0.5,
    )
    risk_ranking = add_top_explanations_to_ranking(
        risk_ranking, prediction_explanations
    )
    actionable_insights = build_actionable_insights(prediction_explanations)
    risk_ranking = add_actionable_insights_to_ranking(
        risk_ranking, actionable_insights
    )
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
        metrics={"sentinel_version": "0.6.0", "models": {}},
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
        risk_ranking=risk_ranking,
        risk_summary=build_risk_summary(
            risk_ranking,
            top_risk=2,
            risk_threshold=0.5,
            selected_model="example",
            evaluation_mode="within_project_temporal_test",
        ),
        prediction_explanations=prediction_explanations,
        explanation_summary=build_explanation_summary(prediction_explanations),
        actionable_insights=actionable_insights,
    )


def test_predict_proba_is_preferred_for_risk_scoring() -> None:
    scores, method = calculate_risk_scores(
        ProbabilityClassifier(), pd.DataFrame({"score": [0.2, 0.9]})
    )

    np.testing.assert_allclose(scores, [0.2, 0.9])
    assert method == "predict_proba"


def test_decision_function_risk_fallback_is_deterministic_and_bounded() -> None:
    features = pd.DataFrame({"margin": [-2.0, 0.0, 2.0]})

    first, method = calculate_risk_scores(DecisionClassifier(), features)
    second, _ = calculate_risk_scores(DecisionClassifier(), features)

    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first, [0.11920292, 0.5, 0.88079708])
    assert method == "decision_function_sigmoid"
    assert np.all((first >= 0.0) & (first <= 1.0))


def test_risk_ranking_has_deterministic_tie_breaks() -> None:
    samples = pd.DataFrame(
        {
            "repository": ["owner/example"] * 3,
            "file_path": ["src/B.java", "src/A.java", "src/C.java"],
            "snapshot_date": ["2022-01-31"] * 3,
            "defect_next_90_days": [0, 1, 0],
        }
    )

    ranking = build_risk_ranking(
        samples,
        [0.8, 0.8, 0.2],
        model="example",
        evaluation_mode="within_project_temporal_test",
        score_method="predict_proba",
        decision_threshold=0.5,
    )

    assert list(ranking.columns) == list(RISK_RANKING_COLUMNS)
    assert ranking["file_path"].tolist() == [
        "src/A.java",
        "src/B.java",
        "src/C.java",
    ]
    assert ranking["rank"].tolist() == [1, 2, 3]
    assert ranking["predicted_label"].tolist() == [1, 1, 0]


def test_report_generation_writes_v5_artifacts_plus_v6_risk_outputs(
    tmp_path: Path,
) -> None:
    paths = save_evaluation_report(sample_report(), tmp_path / "report")

    assert set(paths) == set(REPORT_FILENAMES)
    assert {path.name for path in paths.values()} == set(REPORT_FILENAMES.values())
    assert all(path.exists() for path in paths.values())
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert metrics["experiment_type"] == "within_project"
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert pd.read_csv(paths["model_comparison"]).loc[0, "accuracy"] == 0.75
    assert pd.read_csv(paths["feature_importance"]).loc[0, "rank"] == 1
    ranking = pd.read_csv(paths["risk_ranking"])
    assert list(ranking.columns) == list(RISK_RANKING_COLUMNS)
    assert ranking["risk_score"].between(0, 1).all()
    risk_summary = json.loads(paths["risk_summary"].read_text(encoding="utf-8"))
    assert risk_summary["total_samples"] == 4
    assert risk_summary["high_risk_sample_count"] == 2
    assert len(risk_summary["top_high_risk_samples"]) == 2
    explanations = pd.read_csv(paths["prediction_explanations"])
    assert list(explanations.columns) == list(EXPLANATION_COLUMNS)
    assert len(explanations) == 4
    explanation_summary = json.loads(
        paths["explanation_summary"].read_text(encoding="utf-8")
    )
    assert explanation_summary["total_explained_samples"] == 4
    assert explanation_summary["total_explained_feature_contributions"] == 4
    actionable = json.loads(
        paths["actionable_insights"].read_text(encoding="utf-8")
    )
    assert actionable["schema_version"] == ACTIONABLE_INSIGHTS_SCHEMA_VERSION
    assert actionable["total_samples"] == 4
    assert actionable["feature_definitions"]
    intelligence = json.loads(
        paths["project_intelligence"].read_text(encoding="utf-8")
    )
    priority = pd.read_csv(paths["developer_priority"])
    assert intelligence["schema_version"] == PROJECT_INTELLIGENCE_SCHEMA_VERSION
    assert intelligence["summary"]["total_evaluated_samples"] == 4
    assert intelligence["summary"]["risk_threshold"] == 0.5
    assert list(priority.columns) == list(DEVELOPER_PRIORITY_COLUMNS)
    assert len(priority) == 4
    assert paths["confusion_matrix"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    markdown = paths["summary"].read_text(encoding="utf-8")
    assert "# Sentinel V6 evaluation report" in markdown
    assert "## Key findings" in markdown
    assert "## Actionable risk insights" in markdown
    assert "## Project Risk Intelligence" in markdown


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
            "--top-risk",
            "2",
            "--risk-threshold",
            "0.8",
            "--explain-top-k",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "V6 report directory" in captured.out
    assert captured.err == ""
    assert {path.name for path in destination.iterdir()} == set(
        REPORT_FILENAMES.values()
    )
    ranking = pd.read_csv(destination / "risk_ranking.csv")
    summary = json.loads((destination / "risk_summary.json").read_text())
    assert summary["total_samples"] == len(ranking)
    assert summary["risk_threshold"] == 0.8
    assert len(summary["top_high_risk_samples"]) <= 2
    assert ranking["risk_score"].between(0, 1).all()
    explanations = pd.read_csv(destination / "prediction_explanations.csv")
    explanation_summary = json.loads(
        (destination / "explanation_summary.json").read_text()
    )
    actionable = json.loads(
        (destination / "actionable_insights.json").read_text()
    )
    assert len(explanations) == len(ranking) * 3
    assert explanation_summary["total_explained_samples"] == len(ranking)
    assert actionable["total_samples"] == len(ranking)
    intelligence = json.loads(
        (destination / "project_intelligence.json").read_text()
    )
    priority = pd.read_csv(destination / "developer_priority.csv")
    assert intelligence["summary"]["total_evaluated_samples"] == len(ranking)
    assert intelligence["metadata"]["threshold_semantics"]["risk_threshold"] == 0.8
    assert len(priority) == len(ranking)
    assert all(len(sample["insights"]) <= 3 for sample in actionable["samples"])
    assert all(column in ranking for column in RANKING_INSIGHT_COLUMNS)
    assert ranking["primary_risk_reason"].notna().all()
    assert ranking["recommended_action"].notna().all()
    nonzero = explanations.groupby("sample_id")["absolute_contribution"].sum() > 0
    normalized = explanations.groupby("sample_id")[
        "normalized_contribution"
    ].sum()
    assert np.allclose(normalized.loc[nonzero], 1.0)


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
    assert len(report.risk_ranking) == len(dataset.data)
    assert set(report.risk_ranking["project"]) == {"owner/alpha", "owner/beta"}
    assert report.risk_ranking["rank"].tolist() == list(
        range(1, len(dataset.data) + 1)
    )
    assert report.risk_ranking["risk_score"].between(0, 1).all()
    assert len(report.prediction_explanations) == len(dataset.data) * 8
    assert report.explanation_summary["total_explained_samples"] == len(
        dataset.data
    )
    assert report.actionable_insights["total_samples"] == len(dataset.data)
    assert report.project_intelligence["summary"]["analysis_scope"] == (
        "per_project"
    )
    assert len(report.project_intelligence["projects"]) == 2
    assert report.developer_priority.groupby("project")["rank"].min().eq(1).all()
    assert {
        "top_risk_feature",
        "top_risk_contribution",
        "top_protective_feature",
        "top_protective_contribution",
    } <= set(report.risk_ranking.columns)


def test_cross_project_cli_creates_risk_outputs(tmp_path: Path, capsys) -> None:
    sources: list[Path] = []
    for index, repository in enumerate(("owner/alpha", "owner/beta")):
        source = tmp_path / f"cli-project-{index}.csv"
        make_feature_data(repository, offset=index).to_csv(source, index=False)
        sources.append(source)
    destination = tmp_path / "cross-report"

    exit_code = main(
        [
            "cross-project",
            *(str(source) for source in sources),
            "--output-dir",
            str(destination),
            "--random-state",
            "7",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "V6 report directory" in captured.out
    assert captured.err == ""
    ranking = pd.read_csv(destination / "risk_ranking.csv")
    summary = json.loads((destination / "risk_summary.json").read_text())
    explanations = pd.read_csv(destination / "prediction_explanations.csv")
    explanation_summary = json.loads(
        (destination / "explanation_summary.json").read_text()
    )
    actionable = json.loads(
        (destination / "actionable_insights.json").read_text()
    )
    intelligence = json.loads(
        (destination / "project_intelligence.json").read_text()
    )
    priority = pd.read_csv(destination / "developer_priority.csv")
    assert len(ranking) == 96
    assert summary["total_samples"] == 96
    assert summary["model_metadata"]["evaluation_mode"] == (
        "cross_project_held_out_folds"
    )
    assert len(explanations) == len(ranking) * 8
    assert explanation_summary["total_explained_samples"] == len(ranking)
    assert actionable["total_samples"] == len(ranking)
    assert all(column in ranking for column in RANKING_INSIGHT_COLUMNS)
    assert intelligence["summary"]["analysis_scope"] == "per_project"
    assert {project["project"] for project in intelligence["projects"]} == {
        "owner/alpha",
        "owner/beta",
    }
    assert priority.groupby("project")["rank"].min().eq(1).all()
