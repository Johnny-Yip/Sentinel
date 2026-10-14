from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sentinel.evaluation import (
    CALIBRATION_COLUMNS,
    THRESHOLD_ANALYSIS_COLUMNS,
    build_risk_calibration,
    save_risk_calibration,
)
from sentinel.evaluation.calibration import CALIBRATION_SECTION
from sentinel.evaluation.cli import main
from sentinel.evaluation.experiment import (
    evaluate_cross_project,
    evaluate_within_project,
)
from sentinel.evaluation.reporting import save_evaluation_report
from test_evaluation import sample_report


def ranking(scores, labels, project="owner/alpha") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project": project,
            "risk_score": scores,
            "true_label": labels,
            "score_method": "predict_proba",
        }
    )


def analyze(scores, labels, **kwargs):
    return build_risk_calibration(
        ranking(scores, labels), experiment_type="within_project", **kwargs
    )


def summary(result):
    return result.artifact["projects"][0]


@pytest.mark.parametrize(
    "scores,labels,brier,ece",
    [
        ([0, 1], [0, 1], 0, 0),
        ([1, 0], [0, 1], 1, 1),
        ([0.25] * 4 + [0.75] * 4, [1, 0, 0, 0, 1, 1, 1, 0], 0.1875, 0),
        ([0.9, 0.9, 0.9, 0.9], [0, 0, 0, 1], 0.61, 0.65),
    ],
)
def test_known_calibration_examples(scores, labels, brier, ece):
    project = summary(analyze(scores, labels, calibration_bins=4))
    assert project["brier_score"] == pytest.approx(brier)
    assert project["ece"] == pytest.approx(ece)
    assert project["valid_samples"] == len(scores)


def test_bins_preserve_empty_cells_and_include_boundaries():
    result = analyze([0, 0.5, 0.75, 1], [0, 1, 0, 1], calibration_bins=4)
    assert result.calibration["sample_count"].tolist() == [1, 0, 1, 2]
    assert (
        result.calibration.iloc[1][["mean_predicted_risk", "observed_defect_rate"]]
        .isna()
        .all()
    )
    assert summary(result)["populated_bins"] == 3
    assert summary(result)["ece"] == pytest.approx((0 + 0.5 + 2 * 0.375) / 4)


@pytest.mark.parametrize("labels", [[0, 0], [1, 1], [1]])
def test_single_class_and_small_samples(labels):
    result = analyze([0.7] * len(labels), labels, analysis_thresholds=[0.5, 0.9])
    project = summary(result)
    assert project["brier_score"] == pytest.approx((0.7 - labels[0]) ** 2)
    assert project["ece"] == pytest.approx(abs(0.7 - labels[0]))
    assert project["populated_bins"] == 1
    if labels[0] == 0:
        assert result.threshold_analysis["recall"].isna().all()
        assert project["recommended_threshold"] is None
        assert project["recommendation_status"] == "no_positive_labels"
        assert result.threshold_analysis.iloc[0]["f1"] == 0
        assert pd.isna(result.threshold_analysis.iloc[1]["f1"])
    else:
        assert result.threshold_analysis["specificity"].isna().all()
        assert result.threshold_analysis["false_positive_rate"].isna().all()
        assert project["recommended_threshold"] == 0.5


@pytest.mark.parametrize(
    "frame", [pd.DataFrame(), ranking([], []), pd.DataFrame({"risk_score": [0.5]})]
)
def test_empty_or_missing_inputs_are_unavailable(frame):
    result = build_risk_calibration(
        frame, experiment_type="within_project", calibration_bins=3
    )
    project = summary(result)
    assert project["valid_samples"] == project["populated_bins"] == 0
    for key in (
        "brier_score",
        "ece",
        "recommended_threshold",
        "metrics_at_recommended_threshold",
    ):
        assert project[key] is None
    assert project["recommendation_status"] == "no_valid_samples"
    assert result.calibration["sample_count"].eq(0).all()
    assert (
        result.calibration[["mean_predicted_risk", "observed_defect_rate"]]
        .isna()
        .all()
        .all()
    )
    assert result.threshold_analysis["files_flagged"].eq(0).all()
    assert result.threshold_analysis["percentage_files_flagged"].isna().all()
    json.dumps(result.artifact, allow_nan=False)


def test_invalid_values_are_excluded_without_clipping_imputation_or_label_truncation():
    result = analyze(
        [
            "0.2",
            "0.8",
            None,
            np.nan,
            np.inf,
            -np.inf,
            -0.1,
            1.1,
            "bad",
            pd.NA,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
        ],
        ["0", "1", 0, 1, 0, 1, 0, 1, 0, 1, None, np.inf, "bad", 0.5, 2, pd.NA],
    )
    project = summary(result)
    assert project["valid_samples"] == 2
    assert project["excluded_samples"] == 14
    assert project["invalid_score_samples"] == 8
    assert project["invalid_label_samples"] == 6
    assert project["brier_score"] == pytest.approx(0.04)
    assert (
        result.threshold_analysis.loc[
            result.threshold_analysis.threshold == 0.5, "percentage_files_flagged"
        ].item()
        == 50
    )
    json.dumps(result.artifact, allow_nan=False)


def test_threshold_confusion_metrics_and_inclusive_comparison():
    result = analyze(
        [0.9, 0.8, 0.4, 0.1], [1, 0, 1, 0], analysis_thresholds=[0.5, 0.4, 1, 0]
    )
    rows = result.threshold_analysis.set_index("threshold")
    assert (
        rows.loc[
            0.5, ["precision", "recall", "f1", "specificity", "false_positive_rate"]
        ].tolist()
        == [0.5] * 5
    )
    assert rows.loc[0.5, "files_flagged"] == 2
    assert rows.loc[0.5, "percentage_files_flagged"] == 50
    assert rows.loc[0.4, "f1"] == pytest.approx(0.8)
    assert rows.loc[0.4, "files_flagged"] == 3
    assert rows.loc[0, "files_flagged"] == 4
    assert rows.loc[1, "files_flagged"] == 0
    assert pd.isna(rows.loc[1, "precision"])
    assert rows.loc[1, "recall"] == rows.loc[1, "f1"] == 0
    assert summary(result)["recommended_threshold"] == 0.4
    assert summary(result)["metrics_at_recommended_threshold"]["f1"] == pytest.approx(
        0.8
    )


def test_constant_predictions_and_deterministic_highest_threshold_tie_break():
    result = analyze([0.5, 0.5], [0, 1], analysis_thresholds=[0.5, 0, 0.2, 0.5, 0.9])
    assert result.threshold_analysis.threshold.tolist() == [0, 0.2, 0.5, 0.9]
    assert summary(result)["recommended_threshold"] == 0.5
    assert summary(result)["metrics_at_recommended_threshold"]["f1"] == pytest.approx(
        2 / 3
    )
    assert (
        summary(result)["metrics_at_recommended_threshold"]["percentage_files_flagged"]
        == 100
    )


def test_f1_ties_do_not_use_recall_or_input_order():
    # At 0.9: TP=1, FP=0, FN=2; at 0.1: TP=3, FP=6, FN=0. Both F1=0.5.
    result = analyze(
        [0.9] + [0.1] * 8, [1, 1, 1] + [0] * 6, analysis_thresholds=[0.1, 0.9]
    )
    assert result.threshold_analysis.f1.tolist() == [0.5, 0.5]
    assert summary(result)["recommended_threshold"] == 0.9


def test_row_order_determinism_and_input_preservation():
    frame = ranking([1e-16, 0.3, 0.7, 0.99, 0, 1], [0, 0, 1, 0, 0, 1])
    frame["file_path"] = "Repeated.java"
    original = frame.copy(deep=True)
    first = build_risk_calibration(frame, experiment_type="within_project")
    second = build_risk_calibration(
        frame.sample(frac=1, random_state=21), experiment_type="within_project"
    )
    assert first.artifact == second.artifact
    pd.testing.assert_frame_equal(first.calibration, second.calibration)
    pd.testing.assert_frame_equal(first.threshold_analysis, second.threshold_analysis)
    pd.testing.assert_frame_equal(frame, original)
    assert (
        summary(first)["valid_samples"] == 6
    )  # Snapshot counts, no path deduplication.


def test_cross_project_isolation_under_counterfactual_other_project_changes():
    alpha = ranking([0.9, 0.1], [1, 0])
    beta = ranking([0.9, 0.1], [0, 1], project="owner/beta")
    first = build_risk_calibration(
        pd.concat([alpha, beta]), experiment_type="cross_project"
    )
    altered_beta = ranking([0.5] * 101, [1] * 101, project="owner/beta")
    second = build_risk_calibration(
        pd.concat([altered_beta, alpha]), experiment_type="cross_project"
    )
    isolated = build_risk_calibration(alpha, experiment_type="within_project")
    assert first.artifact["analysis_scope"] == "per_project"
    assert "recommended_threshold" not in first.artifact  # No pooled recommendation.
    assert (
        first.artifact["projects"][0]
        == second.artifact["projects"][0]
        == summary(isolated)
    )
    assert first.artifact["projects"][0]["brier_score"] == pytest.approx(0.01)
    assert first.artifact["projects"][1]["brier_score"] == pytest.approx(0.81)
    for name in ("calibration", "threshold_analysis"):
        expected = getattr(isolated, name)
        for result in (first, second):
            rows = (
                getattr(result, name)
                .query("project == 'owner/alpha'")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(rows, expected)


def test_unknown_cross_project_identity_is_excluded_instead_of_pooled():
    frame = ranking([0, 1, 0.5, 1], [0, 1, 1, 0], project=["known", None, "", np.nan])
    result = build_risk_calibration(frame, experiment_type="cross_project")
    assert result.artifact["excluded_unknown_project_samples"] == 3
    assert result.artifact["project_count"] == 1
    assert summary(result)["valid_samples"] == 1
    assert summary(result)["brier_score"] == 0
    empty = build_risk_calibration(frame.iloc[1:], experiment_type="cross_project")
    assert empty.artifact["projects"] == []
    assert empty.calibration.empty and empty.threshold_analysis.empty
    assert tuple(empty.calibration.columns) == CALIBRATION_COLUMNS


def test_method_provenance_and_missing_columns():
    frame = ranking([0.2, 0.8], [0, 1])
    frame["score_method"] = ["decision_function_sigmoid", None]
    result = build_risk_calibration(frame, experiment_type="within_project")
    assert summary(result)["score_methods"] == ["decision_function_sigmoid"]
    assert summary(result)["unknown_score_method_samples"] == 1
    assert "sigmoid" in " ".join(result.artifact["limitations"])
    result = build_risk_calibration(
        frame.drop(columns="risk_score"), experiment_type="within_project"
    )
    assert summary(result)["valid_samples"] == 0


@pytest.mark.parametrize("bins", [0, -1, 2.5, True, np.nan])
def test_reject_invalid_bins(bins):
    with pytest.raises(ValueError, match="calibration_bins"):
        analyze([0.5], [1], calibration_bins=bins)


@pytest.mark.parametrize(
    "thresholds", [[], [-0.1], [1.1], [np.nan], [np.inf], [None], ["bad"]]
)
def test_reject_invalid_thresholds(thresholds):
    with pytest.raises(ValueError, match="analysis_thresholds"):
        analyze([0.5], [1], analysis_thresholds=thresholds)


@pytest.mark.parametrize("evaluate", [evaluate_within_project, evaluate_cross_project])
def test_invalid_options_rejected_before_training(evaluate, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Phase 7 invalid options must fail before training")

    monkeypatch.setattr("sentinel.evaluation.experiment.train_baselines", forbidden)
    monkeypatch.setattr(
        "sentinel.evaluation.experiment.run_cross_project_evaluation", forbidden
    )
    with pytest.raises(ValueError, match="calibration_bins"):
        evaluate(None, calibration_bins=0)


def reject_constant(value):
    raise AssertionError(f"Nonstandard JSON number: {value}")


def test_artifact_schema_strict_json_csv_blanks_and_idempotent_report(tmp_path):
    result = analyze([0.1, 0.9], [0, 0], calibration_bins=4)
    original = "# Existing V6 report\n\n## Prior section\n\nExisting evidence.\n"
    (tmp_path / "report.md").write_text(original)
    paths = save_risk_calibration(result, tmp_path)
    artifact = json.loads(
        paths["calibration_summary"].read_text(), parse_constant=reject_constant
    )
    assert artifact == result.artifact
    assert artifact["schema_version"] == 1
    assert tuple(pd.read_csv(paths["calibration"]).columns) == CALIBRATION_COLUMNS
    assert (
        tuple(pd.read_csv(paths["threshold_analysis"]).columns)
        == THRESHOLD_ANALYSIS_COLUMNS
    )
    assert ",0,," in paths["calibration"].read_text()
    assert "NaN" not in paths["calibration_summary"].read_text()
    assert "Infinity" not in paths["calibration_summary"].read_text()
    assert paths["summary"].read_text().startswith(original)
    assert "N/A" in paths["summary"].read_text()
    first = {name: path.read_bytes() for name, path in paths.items()}
    save_risk_calibration(result, tmp_path)
    assert first == {name: path.read_bytes() for name, path in paths.items()}
    # Replacing a section preserves unrelated sections after it as well.
    with paths["summary"].open("a") as target:
        target.write("\n## Later section\n\nKeep this evidence.\n")
    save_risk_calibration(result, tmp_path)
    assert (
        paths["summary"]
        .read_text()
        .endswith("## Later section\n\nKeep this evidence.\n")
    )
    assert paths["summary"].read_text().count(CALIBRATION_SECTION) == 1


def test_legacy_report_gets_phase7_without_training_or_changes_to_prior_outputs(
    tmp_path, monkeypatch
):
    report = sample_report()
    original_ranking = report.risk_ranking.copy(deep=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Report generation must use retained predictions")

    monkeypatch.setattr("sentinel.evaluation.experiment.train_baselines", forbidden)
    monkeypatch.setattr(
        "sentinel.evaluation.experiment.run_cross_project_evaluation", forbidden
    )
    paths = save_evaluation_report(report, tmp_path)
    assert report.risk_calibration is None  # Older constructors remain supported.
    pd.testing.assert_frame_equal(report.risk_ranking, original_ranking)
    expected = build_risk_calibration(
        report.risk_ranking, experiment_type=report.experiment_type
    )
    assert json.loads(paths["calibration_summary"].read_text()) == expected.artifact
    markdown = paths["summary"].read_text()
    assert CALIBRATION_SECTION in markdown
    assert "evaluation-derived operating point" in markdown
    assert "optimistic" in markdown


def test_artifact_only_cli_never_trains_and_reproduces_saved_results(
    tmp_path, monkeypatch
):
    source = tmp_path / "risk_ranking.csv"
    ranking([0.12345678901234567, 0.9, 0.5], [0, 1, 0]).to_csv(source, index=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Artifact-only CLI must not evaluate/train a model")

    monkeypatch.setattr("sentinel.evaluation.cli.evaluate_within_project", forbidden)
    monkeypatch.setattr("sentinel.evaluation.cli.evaluate_cross_project", forbidden)
    args = [
        "calibrate",
        str(source),
        "--evaluation-mode",
        "cross_project",
        "--calibration-bins",
        "4",
        "--analysis-thresholds",
        "0.8",
        "0.2",
    ]
    assert main(args) == 0
    artifact = json.loads(
        (tmp_path / "calibration_summary.json").read_text(),
        parse_constant=reject_constant,
    )
    assert artifact["metadata"]["calibration_bins"] == 4
    assert artifact["metadata"]["analysis_thresholds"] == [0.2, 0.8]
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main(args) == 0
    assert first == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_invalid_mode_and_cli_inputs(tmp_path):
    with pytest.raises(ValueError, match="experiment_type"):
        build_risk_calibration(pd.DataFrame(), experiment_type="pooled")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "calibrate",
                "x.csv",
                "--evaluation-mode",
                "within_project",
                "--analysis-thresholds",
                "nan",
            ]
        )
    assert error.value.code == 2
    assert (
        main(
            [
                "calibrate",
                str(tmp_path / "missing.csv"),
                "--evaluation-mode",
                "cross_project",
            ]
        )
        == 1
    )
