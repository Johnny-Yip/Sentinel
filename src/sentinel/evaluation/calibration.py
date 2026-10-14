"""Retrospective calibration and threshold diagnostics from retained risk rows.

This module never fits, predicts, or changes a model's selected threshold.
All statistics and candidate comparisons are local to one evaluation project.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


CALIBRATION_SCHEMA_VERSION = 1
DEFAULT_CALIBRATION_BINS = 10
DEFAULT_ANALYSIS_THRESHOLDS = tuple(index / 20 for index in range(21))
CALIBRATION_COLUMNS = (
    "project",
    "bin_lower",
    "bin_upper",
    "sample_count",
    "mean_predicted_risk",
    "observed_defect_rate",
)
THRESHOLD_ANALYSIS_COLUMNS = (
    "project",
    "threshold",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_positive_rate",
    "files_flagged",
    "percentage_files_flagged",
)
CALIBRATION_SECTION = "## Risk Calibration & Operating Threshold"
_LIMITATIONS = (
    "The recommended threshold is a retrospective, evaluation-derived operating "
    "point maximizing F1 among the evaluated candidates, not a universally optimal "
    "threshold. Its metrics reuse the evaluation labels and are optimistic for "
    "deployment; validate any operational choice on independent future data.",
    "Calibration compares predicted risk with the observed frequency of the "
    "heuristic defect label. Small bins, single-class samples, repeated file "
    "snapshots and overlapping label windows limit statistical interpretation; "
    "no confidence intervals or causal claims are provided.",
    "Bounded scores (including sigmoid decision margins) can be assessed as risk "
    "estimates, but these diagnostics do not turn them into calibrated "
    "probabilities or recalibrate a model. Brier score also reflects discrimination; "
    "ECE depends on binning and can hide within-bin errors.",
)


@dataclass
class RiskCalibrationResult:
    """JSON-ready project summaries and the two complete diagnostic tables."""

    artifact: dict[str, Any]
    calibration: pd.DataFrame
    threshold_analysis: pd.DataFrame


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def validate_calibration_options(
    bins: int,
    thresholds: Sequence[float] | None,
) -> tuple[int, tuple[float, ...]]:
    """Validate before training when used by the existing experiment adapters."""
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins <= 0:
        raise ValueError("calibration_bins must be a positive integer.")
    candidates = [
        _number(value)
        for value in (DEFAULT_ANALYSIS_THRESHOLDS if thresholds is None else thresholds)
    ]
    if not candidates or any(
        value is None or not 0 <= value <= 1 for value in candidates
    ):
        raise ValueError(
            "analysis_thresholds must be nonempty finite values in [0, 1]."
        )
    return int(bins), tuple(sorted(set(candidates)))


def _project(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    # Do not normalize known names into each other or guess missing identities.
    return value if value.strip() else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _analyze_project(
    project: str | None,
    records: list[dict[str, Any]],
    bins: int,
    thresholds: tuple[float, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[tuple[float, int]] = []
    invalid_scores = invalid_labels = 0
    methods: set[str] = set()
    unknown_methods = 0
    for row in records:
        score = _number(row.get("risk_score"))
        label = _number(row.get("true_label"))
        valid_score = score is not None and 0 <= score <= 1
        valid_label = label in (0.0, 1.0)
        invalid_scores += int(not valid_score)
        invalid_labels += int(not valid_label)
        if valid_score and valid_label:
            pairs.append((score, int(label)))
            method = _project(row.get("score_method"))
            if method is None:
                unknown_methods += 1
            else:
                methods.add(method)
    # Canonical order and accurate summation make artifacts independent of row order.
    pairs.sort()
    scores = np.array([pair[0] for pair in pairs], dtype=float)
    labels = np.array([pair[1] for pair in pairs], dtype=int)
    count = len(pairs)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Internal edges belong to the bin on their right; 1 belongs to the last bin.
    assignments = np.searchsorted(edges[1:-1], scores, side="right")
    calibration = []
    for index in range(bins):
        selected = assignments == index
        size = int(selected.sum())
        calibration.append(
            {
                "project": project,
                "bin_lower": float(edges[index]),
                "bin_upper": float(edges[index + 1]),
                "sample_count": size,
                "mean_predicted_risk": (
                    math.fsum(scores[selected]) / size if size else None
                ),
                "observed_defect_rate": int(labels[selected].sum()) / size
                if size
                else None,
            }
        )
    populated = [row for row in calibration if row["sample_count"]]
    analysis = []
    for threshold in thresholds:
        flagged = scores >= threshold
        tp = int(((labels == 1) & flagged).sum())
        fp = int(((labels == 0) & flagged).sum())
        fn = int(((labels == 1) & ~flagged).sum())
        tn = int(((labels == 0) & ~flagged).sum())
        analysis.append(
            {
                "project": project,
                "threshold": threshold,
                "precision": _ratio(tp, tp + fp),
                "recall": _ratio(tp, tp + fn),
                "f1": _ratio(2 * tp, 2 * tp + fp + fn),
                "specificity": _ratio(tn, tn + fp),
                "false_positive_rate": _ratio(fp, tn + fp),
                "files_flagged": tp + fp,
                "percentage_files_flagged": 100 * (tp + fp) / count if count else None,
            }
        )
    positives = int(labels.sum())
    # With no positives there is no observed positive-class retrieval evidence.
    recommended = (
        max(analysis, key=lambda row: (row["f1"], row["threshold"]))
        if positives
        else None
    )
    summary = {
        "project": project,
        "total_samples": len(records),
        "valid_samples": count,
        "excluded_samples": len(records) - count,
        "invalid_score_samples": invalid_scores,
        "invalid_label_samples": invalid_labels,
        "positive_labels": positives,
        "negative_labels": count - positives,
        "populated_bins": len(populated),
        "brier_score": math.fsum((score - label) ** 2 for score, label in pairs) / count
        if count
        else None,
        "ece": math.fsum(
            row["sample_count"]
            / count
            * abs(row["mean_predicted_risk"] - row["observed_defect_rate"])
            for row in populated
        )
        if count
        else None,
        "recommended_threshold": recommended["threshold"] if recommended else None,
        "recommendation_criterion": "maximum_f1",
        "recommendation_status": (
            "available"
            if recommended
            else "no_valid_samples"
            if not count
            else "no_positive_labels"
        ),
        "metrics_at_recommended_threshold": (
            {key: value for key, value in recommended.items() if key != "project"}
            if recommended
            else None
        ),
        "score_methods": sorted(methods),
        "unknown_score_method_samples": unknown_methods,
    }
    return summary, calibration, analysis


def build_risk_calibration(
    ranking: pd.DataFrame,
    *,
    experiment_type: str,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
    analysis_thresholds: Sequence[float] | None = None,
) -> RiskCalibrationResult:
    """Analyze existing risk_score/true_label pairs without model access.

    Each row is a file snapshot, not a unique file. The same complete-case
    population supplies calibration, classification metrics and flagged counts.
    Missing projects in cross-project inputs are excluded, never pooled.
    """
    bins, thresholds = validate_calibration_options(
        calibration_bins, analysis_thresholds
    )
    if experiment_type not in {"within_project", "cross_project"}:
        raise ValueError("experiment_type must be within_project or cross_project.")
    groups: dict[str | None, list[dict[str, Any]]] = {}
    excluded_projects = 0
    for row in ranking.to_dict("records"):
        project = _project(row.get("project"))
        if project is None and experiment_type == "cross_project":
            excluded_projects += 1
            continue
        groups.setdefault(project, []).append(row)
    if not groups and experiment_type == "within_project":
        groups[None] = []
    projects, calibration, analysis = [], [], []
    for project in sorted(groups, key=lambda name: (name is not None, name or "")):
        summary, reliability_rows, threshold_rows = _analyze_project(
            project, groups[project], bins, thresholds
        )
        projects.append(summary)
        calibration.extend(reliability_rows)
        analysis.extend(threshold_rows)
    artifact = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "experiment_type": experiment_type,
        "analysis_scope": "per_project"
        if experiment_type == "cross_project" or len(projects) > 1
        else "single_project",
        "project_count": len(projects),
        "total_samples": len(ranking),
        "excluded_unknown_project_samples": excluded_projects,
        "projects": projects,
        "metadata": {
            "input_dependencies": ["risk_ranking"],
            "sample_unit": "evaluated_file_snapshot",
            "calibration_bins": bins,
            "bin_edges": np.linspace(0.0, 1.0, bins + 1).tolist(),
            "binning": "equal_width; [lower, upper), except final bin includes 1",
            "analysis_thresholds": list(thresholds),
            "threshold_comparison": "risk_score >= threshold",
            "recommendation_criterion": "maximum_f1",
            "tie_break": "highest_threshold_among_exact_f1_ties",
            "recommendation_scope": "retrospective_evaluation_only; does not change validation-selected thresholds or predictions",
            "missing_values": "null in JSON, blank in CSV, N/A in Markdown; zero denominators are undefined",
            "valid_samples": "finite risk_score in [0,1] and binary true_label; numeric strings accepted; no clipping or imputation",
            "flagged_population": "valid labeled file snapshots only; repeated paths count separately",
            "excluded_counts": "invalid score and label counts may overlap; unknown cross-project identities excluded before metric analysis",
            "brier_formula": "mean((risk_score - true_label)^2)",
            "ece_formula": "sum(bin_sample_count / valid_samples * abs(mean_predicted_risk - observed_defect_rate))",
        },
        "limitations": list(_LIMITATIONS),
    }
    return RiskCalibrationResult(
        artifact,
        pd.DataFrame(calibration, columns=CALIBRATION_COLUMNS),
        pd.DataFrame(analysis, columns=THRESHOLD_ANALYSIS_COLUMNS),
    )


def calibration_markdown_lines(artifact: dict[str, Any]) -> list[str]:
    """Render the same project summaries saved in calibration_summary.json."""

    def metric(value: Any) -> str:
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        CALIBRATION_SECTION,
        "",
        "Calibration asks whether files assigned a risk near 0.7 show the defect "
        "label about 70% of the time. Brier score is mean squared risk error; "
        "ECE is the sample-weighted absolute gap between mean risk and observed "
        "defect rate within bins. Lower values are better, subject to the limitations below.",
        "",
        "Each project is analyzed separately. Flagged counts and percentages refer "
        "to valid labeled file snapshots; repeated paths count separately. "
        "Precision measures how many flagged samples are positive; recall measures "
        "how many positives are found. F1 balances the two. Higher cutoffs flag fewer "
        "or equal samples and cannot increase recall; precision need not improve.",
        "",
        "| Project | Valid samples | Populated bins | Brier | ECE | Recommended threshold | Precision | Recall | F1 | Files flagged | Flagged % |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for project in artifact["projects"]:
        metrics = project["metrics_at_recommended_threshold"] or {}
        name = project["project"]
        name = (
            name.replace("|", "\\|").replace("\n", " ").replace("\r", " ")
            if name
            else "unknown"
        )
        cells = [
            name,
            str(project["valid_samples"]),
            str(project["populated_bins"]),
            metric(project["brier_score"]),
            metric(project["ece"]),
            metric(project["recommended_threshold"]),
        ]
        cells.extend(metric(metrics.get(key)) for key in ("precision", "recall", "f1"))
        cells.append(str(metrics["files_flagged"]) if metrics else "N/A")
        cells.append(metric(metrics.get("percentage_files_flagged")))
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Recommendation: maximum F1 among the configured candidates; exact ties "
            "choose the highest threshold. No recommendation is available without "
            "valid samples containing at least one positive label. Undefined rates "
            "are N/A. Existing model predictions and validation-selected thresholds are unchanged.",
            "",
            f"Rows excluded for unknown cross-project identity: {artifact['excluded_unknown_project_samples']}.",
            "",
            *[f"- {limitation}" for limitation in artifact["limitations"]],
            "",
            "See `calibration.csv`, `threshold_analysis.csv`, and "
            "`calibration_summary.json` for bin counts, all candidates and exclusions.",
            "",
        ]
    )
    return lines
