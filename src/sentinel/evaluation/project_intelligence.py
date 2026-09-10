"""Deterministic project-level intelligence from existing V6 risk outputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from sentinel.evaluation.insights import FEATURE_INSIGHTS


PROJECT_INTELLIGENCE_SCHEMA_VERSION = 1
PRIORITY_RISK_WEIGHT = 0.80
PRIORITY_HOTSPOT_WEIGHT = 0.20
RISK_DISTRIBUTION_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEVELOPER_PRIORITY_COLUMNS = (
    "rank",
    "project",
    "identifier",
    "snapshot_date",
    "predicted_risk",
    "primary_risk_reason",
    "recommended_action",
    "risk_signal_count",
    "recurring_high_risk_count",
    "recurring_high_risk_rate",
    "recurring_hotspot_evidence",
    "priority_score",
)

_PRIORITY_FORMULA = (
    "priority_score = 0.80 * predicted_risk + 0.20 * "
    "recurring_hotspot_evidence; recurring_hotspot_evidence = "
    "high_risk_rate * min(high_risk_count / 2, 1)"
)
_LIMITATIONS = [
    "Project intelligence summarizes fitted-model signals and does not prove that "
    "an evaluated entity contains a defect.",
    "Risk signals are fitted-model associations, not causal claims.",
    "Priority scores are deterministic inspection heuristics, not probabilities.",
    "Raw contribution magnitudes are only comparable within the same explanation "
    "method and contribution space.",
    "Temporal changes are descriptive; Sentinel does not claim statistical "
    "significance.",
]


@dataclass(frozen=True)
class ProjectIntelligenceResult:
    """Machine-readable intelligence plus its complete inspection queue."""

    artifact: dict[str, Any]
    developer_priority: pd.DataFrame


def _percentage(numerator: float, denominator: float) -> float:
    return float(numerator / denominator * 100.0) if denominator else 0.0


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str).str.strip()


def _optional_text(value: Any, fallback: str = "") -> str:
    return fallback if value is None or pd.isna(value) else str(value)


def _validate_inputs(ranking: pd.DataFrame, risk_threshold: float) -> None:
    if not np.isfinite(risk_threshold) or not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("risk_threshold must be between 0 and 1.")
    if ranking.empty:
        return
    if "risk_score" not in ranking:
        raise ValueError("Project intelligence requires a risk_score column.")
    scores = pd.to_numeric(ranking["risk_score"], errors="coerce").to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Project intelligence risk scores must be within [0, 1].")


def _risk_distribution(scores: pd.Series) -> list[dict[str, Any]]:
    values = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    counts, _ = np.histogram(values, bins=RISK_DISTRIBUTION_EDGES)
    total = len(values)
    result: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        lower = RISK_DISTRIBUTION_EDGES[index]
        upper = RISK_DISTRIBUTION_EDGES[index + 1]
        inclusive_upper = index == len(counts) - 1
        result.append(
            {
                "range": (
                    f"[{lower:.1f}, {upper:.1f}]"
                    if inclusive_upper
                    else f"[{lower:.1f}, {upper:.1f})"
                ),
                "minimum": lower,
                "maximum": upper,
                "inclusive_maximum": inclusive_upper,
                "sample_count": int(count),
                "percentage_of_samples": _percentage(int(count), total),
            }
        )
    return result


def calculate_risk_concentration(scores: pd.Series | list[float]) -> dict[str, Any]:
    """Calculate deterministic concentration metrics for ordered risk mass.

    Top-fraction sample counts use ``ceil(n * fraction)``. The 50% metric is the
    smallest number of descending scores whose cumulative sum reaches at least
    half of total predicted risk. All shares and the 50% count are zero when
    total predicted risk is zero.
    """
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Concentration scores must be finite and non-negative.")
    ordered = np.sort(values)[::-1]
    total = float(ordered.sum())
    sample_count = len(ordered)

    def top_share(fraction: float) -> tuple[int, float]:
        count = ceil(sample_count * fraction) if sample_count else 0
        contribution = float(ordered[:count].sum()) if total else 0.0
        return count, _percentage(contribution, total)

    top_10_count, top_10_share = top_share(0.10)
    top_20_count, top_20_share = top_share(0.20)
    if total:
        samples_for_half = int(
            np.searchsorted(np.cumsum(ordered), total * 0.50, side="left") + 1
        )
    else:
        samples_for_half = 0
    return {
        "aggregate_predicted_risk": total,
        "top_10_percent_sample_count": top_10_count,
        "top_10_percent_risk_contribution_percentage": top_10_share,
        "top_20_percent_sample_count": top_20_count,
        "top_20_percent_risk_contribution_percentage": top_20_share,
        "samples_responsible_for_50_percent_of_predicted_risk": samples_for_half,
    }


def _supported_explanations(explanations: pd.DataFrame) -> pd.DataFrame:
    required = {"feature", "contribution", "direction"}
    if explanations.empty or not required <= set(explanations.columns):
        return pd.DataFrame(columns=list(explanations.columns))
    return explanations.loc[
        explanations["feature"].astype(str).isin(FEATURE_INSIGHTS)
    ].copy()


def _signal_records(
    explanations: pd.DataFrame,
    *,
    direction: str,
    total_samples: int,
) -> list[dict[str, Any]]:
    supported = _supported_explanations(explanations)
    if supported.empty:
        return []
    selected = supported.loc[supported["direction"] == direction].copy()
    if selected.empty:
        return []
    selected["contribution"] = pd.to_numeric(
        selected["contribution"], errors="coerce"
    )
    selected = selected.loc[np.isfinite(selected["contribution"])]
    rows: list[dict[str, Any]] = []
    for feature, group in selected.groupby("feature", sort=True):
        contributions = group["contribution"].astype(float)
        sample_key_columns = [
            column
            for column in ("sample_id", "model", "evaluation_mode")
            if column in group
        ]
        occurrence_count = (
            len(group[sample_key_columns].drop_duplicates())
            if sample_key_columns
            else len(group)
        )
        rows.append(
            {
                "feature": str(feature),
                "category": FEATURE_INSIGHTS[str(feature)].category,
                "occurrence_count": occurrence_count,
                "percentage_of_samples": _percentage(
                    occurrence_count, total_samples
                ),
                "aggregate_contribution": float(contributions.sum()),
                "mean_contribution": float(contributions.mean()),
                "contribution_spaces": (
                    sorted(group["contribution_space"].astype(str).unique().tolist())
                    if "contribution_space" in group
                    else []
                ),
                "explanation_methods": (
                    sorted(group["explanation_method"].astype(str).unique().tolist())
                    if "explanation_method" in group
                    else []
                ),
            }
        )
    return rows


def _signal_profile(
    explanations: pd.DataFrame, total_samples: int
) -> dict[str, Any]:
    risk = _signal_records(
        explanations, direction="increases_risk", total_samples=total_samples
    )
    protective = _signal_records(
        explanations, direction="decreases_risk", total_samples=total_samples
    )
    common_risk = sorted(
        risk,
        key=lambda row: (
            -row["occurrence_count"],
            -abs(row["aggregate_contribution"]),
            row["feature"],
        ),
    )
    common_protective = sorted(
        protective,
        key=lambda row: (
            -row["occurrence_count"],
            -abs(row["aggregate_contribution"]),
            row["feature"],
        ),
    )
    strongest_risk = sorted(
        risk,
        key=lambda row: (-row["aggregate_contribution"], row["feature"]),
    )
    strongest_protective = sorted(
        protective,
        key=lambda row: (row["aggregate_contribution"], row["feature"]),
    )
    spaces = (
        sorted(explanations["contribution_space"].astype(str).unique().tolist())
        if not explanations.empty and "contribution_space" in explanations
        else []
    )
    methods = (
        sorted(explanations["explanation_method"].astype(str).unique().tolist())
        if not explanations.empty and "explanation_method" in explanations
        else []
    )
    return {
        "supported": not explanations.empty,
        "most_common_risk_signals": common_risk,
        "most_common_protective_signals": common_protective,
        "strongest_aggregate_risk_contributors": strongest_risk,
        "strongest_aggregate_protective_contributors": strongest_protective,
        "contribution_spaces": spaces,
        "explanation_methods": methods,
        "aggregate_contributions_comparable": len(spaces) <= 1,
    }


def _entity_statistics(
    ranking: pd.DataFrame, risk_threshold: float
) -> tuple[dict[str, dict[str, Any]], bool, float]:
    identifiers = _text_series(ranking, "file_path")
    available = identifiers.ne("")
    coverage = _percentage(int(available.sum()), len(ranking))
    statistics: dict[str, dict[str, Any]] = {}
    working = ranking.loc[available].copy()
    if working.empty:
        return statistics, False, coverage
    working["_identifier"] = identifiers.loc[available]
    working["_risk_score"] = pd.to_numeric(working["risk_score"])
    working["_high_risk"] = working["_risk_score"] >= risk_threshold
    for identifier, group in working.groupby("_identifier", sort=True):
        high_risk_count = int(group["_high_risk"].sum())
        statistics[str(identifier)] = {
            "observation_count": len(group),
            "high_risk_count": high_risk_count,
            "high_risk_rate": high_risk_count / len(group),
            "mean_risk": float(group["_risk_score"].mean()),
            "max_risk": float(group["_risk_score"].max()),
            "aggregate_predicted_risk": float(group["_risk_score"].sum()),
        }
    return statistics, True, coverage


def _identifier_explanations(
    explanations: pd.DataFrame, identifier: str
) -> pd.DataFrame:
    if explanations.empty or "file_path" not in explanations:
        return explanations.iloc[0:0].copy()
    return explanations.loc[_text_series(explanations, "file_path") == identifier]


def _build_hotspots(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    risk_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    statistics, identifiers_available, coverage = _entity_statistics(
        ranking, risk_threshold
    )
    total_predicted_risk = float(
        pd.to_numeric(ranking.get("risk_score", pd.Series(dtype=float))).sum()
    )
    repeated = {
        identifier: values
        for identifier, values in statistics.items()
        if values["high_risk_count"] >= 2
    }
    hotspots: list[dict[str, Any]] = []
    for identifier, values in repeated.items():
        signals = _signal_records(
            _identifier_explanations(explanations, identifier),
            direction="increases_risk",
            total_samples=values["observation_count"],
        )
        dominant = sorted(
            signals,
            key=lambda row: (
                -row["occurrence_count"],
                -abs(row["aggregate_contribution"]),
                row["feature"],
            ),
        )[:3]
        actions = [
            FEATURE_INSIGHTS[row["feature"]].risk_recommendation
            for row in dominant
        ]
        hotspots.append(
            {
                "identifier": identifier,
                "identifier_type": "file_path",
                "observation_count": values["observation_count"],
                "high_risk_count": values["high_risk_count"],
                "high_risk_rate": values["high_risk_rate"],
                "mean_risk": values["mean_risk"],
                "max_risk": values["max_risk"],
                "dominant_risk_signals": dominant,
                "recommended_inspection_actions": actions,
            }
        )
    hotspots.sort(
        key=lambda row: (
            -row["high_risk_count"],
            -row["high_risk_rate"],
            -row["mean_risk"],
            -row["max_risk"],
            row["identifier"],
        )
    )

    repeated_risk = sum(
        values["aggregate_predicted_risk"] for values in repeated.values()
    )
    repeated_high_risk_count = sum(
        values["high_risk_count"] for values in repeated.values()
    )
    high_risk_total = sum(
        values["high_risk_count"] for values in statistics.values()
    )
    concentration = {
        "supported": identifiers_available,
        "identifier_type": "file_path" if identifiers_available else None,
        "identifier_coverage_percentage": coverage,
        "repeated_high_risk_entity_definition": (
            "an identifier with at least two observations at or above the "
            "configured risk threshold"
        ),
        "repeated_high_risk_entity_count": len(repeated),
        "repeated_entity_predicted_risk_contribution_percentage": _percentage(
            repeated_risk, total_predicted_risk
        ),
        "repeated_entity_high_risk_sample_percentage": _percentage(
            repeated_high_risk_count, high_risk_total
        ),
    }
    return hotspots, concentration


def _priority_queue(
    ranking: pd.DataFrame,
    entity_statistics: dict[str, dict[str, Any]],
    project: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_order, row in enumerate(ranking.itertuples(index=False)):
        identifier = _optional_text(getattr(row, "file_path", ""))
        stats = entity_statistics.get(identifier)
        high_risk_count = int(stats["high_risk_count"]) if stats else 0
        high_risk_rate = float(stats["high_risk_rate"]) if stats else 0.0
        hotspot_evidence = high_risk_rate * min(high_risk_count / 2.0, 1.0)
        predicted_risk = float(getattr(row, "risk_score"))
        priority_score = (
            PRIORITY_RISK_WEIGHT * predicted_risk
            + PRIORITY_HOTSPOT_WEIGHT * hotspot_evidence
        )
        risk_signal_value = getattr(row, "risk_signal_count", 0)
        risk_signal_count = (
            0 if pd.isna(risk_signal_value) else int(risk_signal_value)
        )
        rows.append(
            {
                "project": project or "",
                "identifier": identifier,
                "snapshot_date": _optional_text(
                    getattr(row, "snapshot_date", "")
                ),
                "predicted_risk": predicted_risk,
                "primary_risk_reason": _optional_text(
                    getattr(row, "primary_risk_reason", None),
                    "No actionable explanation was available.",
                ),
                "recommended_action": _optional_text(
                    getattr(row, "recommended_action", None),
                    "Review the detailed prediction output manually.",
                ),
                "risk_signal_count": risk_signal_count,
                "recurring_high_risk_count": high_risk_count,
                "recurring_high_risk_rate": high_risk_rate,
                "recurring_hotspot_evidence": hotspot_evidence,
                "priority_score": priority_score,
                "_source_order": source_order,
            }
        )
    if not rows:
        return pd.DataFrame(columns=DEVELOPER_PRIORITY_COLUMNS)
    queue = pd.DataFrame(rows).sort_values(
        [
            "priority_score",
            "predicted_risk",
            "recurring_high_risk_count",
            "identifier",
            "snapshot_date",
            "_source_order",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    queue.insert(0, "rank", np.arange(1, len(queue) + 1))
    return queue.loc[:, DEVELOPER_PRIORITY_COLUMNS]


def _temporal_analysis(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    risk_threshold: float,
) -> dict[str, Any]:
    if ranking.empty or "snapshot_date" not in ranking:
        return {
            "supported": False,
            "period_granularity": None,
            "periods": [],
            "direction": "not_available",
            "unsupported_reason": "No evaluated timestamps were available.",
        }
    timestamps = pd.to_datetime(ranking["snapshot_date"], errors="coerce")
    if timestamps.isna().any():
        return {
            "supported": False,
            "period_granularity": None,
            "periods": [],
            "direction": "not_available",
            "unsupported_reason": (
                "Temporal analysis requires a valid timestamp for every evaluated "
                "sample."
            ),
        }
    working = ranking.copy()
    working["_period"] = timestamps.dt.to_period("M").astype(str)
    explanation_periods = explanations.copy()
    if not explanations.empty and "snapshot_date" in explanations:
        explanation_timestamps = pd.to_datetime(
            explanations["snapshot_date"], errors="coerce"
        )
        explanation_periods = explanations.loc[
            explanation_timestamps.notna()
        ].copy()
        explanation_periods["_period"] = explanation_timestamps.loc[
            explanation_timestamps.notna()
        ].dt.to_period("M").astype(str)

    periods: list[dict[str, Any]] = []
    previous_mean: float | None = None
    for period, group in working.groupby("_period", sort=True):
        scores = pd.to_numeric(group["risk_score"]).astype(float)
        high_risk_count = int((scores >= risk_threshold).sum())
        dominant: str | None = None
        dominant_category: str | None = None
        if "_period" in explanation_periods:
            signals = _signal_records(
                explanation_periods.loc[explanation_periods["_period"] == period],
                direction="increases_risk",
                total_samples=len(group),
            )
            if signals:
                top = sorted(
                    signals,
                    key=lambda row: (
                        -row["occurrence_count"],
                        -abs(row["aggregate_contribution"]),
                        row["feature"],
                    ),
                )[0]
                dominant = top["feature"]
                dominant_category = top["category"]
        mean_risk = float(scores.mean())
        periods.append(
            {
                "period": str(period),
                "sample_count": len(group),
                "mean_risk": mean_risk,
                "high_risk_count": high_risk_count,
                "high_risk_rate": high_risk_count / len(group),
                "dominant_risk_signal": dominant,
                "dominant_risk_signal_category": dominant_category,
                "change_in_mean_risk_from_previous_period": (
                    None if previous_mean is None else mean_risk - previous_mean
                ),
            }
        )
        previous_mean = mean_risk
    if len(periods) < 2:
        direction = "not_available"
    else:
        change = periods[-1]["mean_risk"] - periods[0]["mean_risk"]
        direction = (
            "increasing"
            if change > 0.0
            else "decreasing"
            if change < 0.0
            else "stable"
        )
    return {
        "supported": True,
        "period_granularity": "calendar_month",
        "periods": periods,
        "direction": direction,
        "unsupported_reason": None,
    }


def _scope_analysis(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    *,
    project: str | None,
    risk_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scores = pd.to_numeric(
        ranking.get("risk_score", pd.Series(dtype=float)), errors="coerce"
    ).astype(float)
    high_risk = scores >= risk_threshold
    signal_profile = _signal_profile(explanations, len(ranking))
    hotspots, repeated_concentration = _build_hotspots(
        ranking, explanations, risk_threshold
    )
    entity_statistics, identifiers_available, identifier_coverage = (
        _entity_statistics(ranking, risk_threshold)
    )
    concentration = calculate_risk_concentration(scores.tolist())
    concentration["repeated_high_risk_entities"] = repeated_concentration
    temporal = _temporal_analysis(ranking, explanations, risk_threshold)
    queue = _priority_queue(ranking, entity_statistics, project)
    summary = {
        "project": project,
        "total_evaluated_samples": len(ranking),
        "high_risk_sample_count": int(high_risk.sum()),
        "high_risk_percentage": _percentage(int(high_risk.sum()), len(ranking)),
        "risk_threshold": risk_threshold,
        "mean_predicted_risk": float(scores.mean()) if len(scores) else None,
        "median_predicted_risk": float(scores.median()) if len(scores) else None,
        "maximum_predicted_risk": float(scores.max()) if len(scores) else None,
        "risk_score_distribution": _risk_distribution(scores),
        "dominant_risk_signals": signal_profile[
            "most_common_risk_signals"
        ][:3],
        "dominant_protective_signals": signal_profile[
            "most_common_protective_signals"
        ][:3],
        "unique_risky_entities": sum(
            values["high_risk_count"] > 0 for values in entity_statistics.values()
        ),
        "identifier_type": "file_path" if identifiers_available else None,
        "identifier_coverage_percentage": identifier_coverage,
    }
    developer_priority = {
        "queue_size": len(queue),
        "priority_score_formula": _PRIORITY_FORMULA,
        "priority_score_is_probability": False,
        "top_priorities": queue.head(5).to_dict(orient="records"),
    }
    return (
        {
            "project": project,
            "summary": summary,
            "risk_concentration": concentration,
            "hotspots": hotspots,
            "risk_signal_profile": signal_profile,
            "temporal_analysis": temporal,
            "developer_priority": developer_priority,
        },
        queue,
    )


def _unsupported_features(
    explanations: pd.DataFrame, actionable_insights: dict[str, Any]
) -> list[str]:
    features = (
        set(explanations["feature"].dropna().astype(str))
        if "feature" in explanations
        else set()
    )
    features -= set(FEATURE_INSIGHTS)
    metadata = actionable_insights.get("insight_metadata", {})
    features.update(str(value) for value in metadata.get("unsupported_features", []))
    return sorted(features)


def _metadata(
    *,
    risk_threshold: float,
    analyses: list[dict[str, Any]],
    unsupported_features: list[str],
    boundary_mode: str,
) -> dict[str, Any]:
    temporal_available = bool(analyses) and all(
        analysis["temporal_analysis"]["supported"] for analysis in analyses
    )
    identifiers_available = bool(analyses) and all(
        analysis["summary"]["identifier_type"] is not None for analysis in analyses
    )
    unsupported: list[str] = []
    if not temporal_available:
        unsupported.append("temporal_analysis")
    if not identifiers_available:
        unsupported.append("recurring_hotspots")
    if analyses and not all(
        analysis["risk_signal_profile"]["supported"] for analysis in analyses
    ):
        unsupported.append("risk_signal_profile")
    return {
        "deterministic": True,
        "project_boundary_mode": boundary_mode,
        "tie_breaking_rules": {
            "hotspots": (
                "high_risk_count_desc, high_risk_rate_desc, mean_risk_desc, "
                "max_risk_desc, identifier_asc"
            ),
            "signals": (
                "occurrence_count_desc, absolute_aggregate_contribution_desc, "
                "feature_asc"
            ),
            "developer_priority": (
                "priority_score_desc, predicted_risk_desc, "
                "recurring_high_risk_count_desc, identifier_asc, "
                "snapshot_date_asc, source_order_asc"
            ),
        },
        "threshold_semantics": {
            "risk_threshold": risk_threshold,
            "comparison": "risk_score >= risk_threshold",
            "source": "the existing evaluation --risk-threshold setting",
            "note": (
                "This cutoff summarizes high-risk samples and does not replace "
                "the validation-selected prediction threshold."
            ),
        },
        "definitions": {
            "top_10_percent_concentration": (
                "sum of the highest ceil(10% * sample_count) risk scores divided "
                "by total predicted risk"
            ),
            "top_20_percent_concentration": (
                "sum of the highest ceil(20% * sample_count) risk scores divided "
                "by total predicted risk"
            ),
            "samples_for_50_percent": (
                "smallest descending-score prefix whose sum reaches at least 50% "
                "of total predicted risk; zero when total predicted risk is zero"
            ),
            "recurring_high_risk_entity": (
                "an available file_path with at least two observations at or above "
                "the configured risk threshold"
            ),
            "risk_distribution": (
                "fixed left-closed bands [0.0,0.2), [0.2,0.4), [0.4,0.6), "
                "[0.6,0.8), and [0.8,1.0]"
            ),
            "priority_score": _PRIORITY_FORMULA,
        },
        "unsupported_analyses": unsupported,
        "unknown_or_unsupported_features": unsupported_features,
        "temporal_analysis_available": temporal_available,
        "hotspot_identifiers_available": identifiers_available,
        "limitations": list(_LIMITATIONS),
    }


def build_project_intelligence(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    actionable_insights: dict[str, Any],
    *,
    risk_threshold: float,
    experiment_type: str,
) -> ProjectIntelligenceResult:
    """Aggregate existing V6 outputs without recomputing model explanations."""
    _validate_inputs(ranking, risk_threshold)
    working = ranking.copy()
    working["_project"] = _text_series(working, "project")
    explanation_working = explanations.copy()
    explanation_working["_project"] = _text_series(
        explanation_working, "project"
    )
    all_project_values = sorted(working["_project"].unique().tolist())
    projects = [project for project in all_project_values if project]
    preserve_boundaries = experiment_type == "cross_project" or len(projects) > 1

    analyses: list[dict[str, Any]] = []
    queues: list[pd.DataFrame] = []
    if preserve_boundaries:
        project_values = all_project_values
        for project in project_values:
            project_ranking = working.loc[working["_project"] == project].drop(
                columns="_project"
            )
            project_explanations = explanation_working.loc[
                explanation_working["_project"] == project
            ].drop(columns="_project")
            analysis, queue = _scope_analysis(
                project_ranking,
                project_explanations,
                project=project or None,
                risk_threshold=risk_threshold,
            )
            analyses.append(analysis)
            queues.append(queue)
        summary: dict[str, Any] = {
            "analysis_scope": "per_project",
            "project_count": len(analyses),
            "projects": [analysis["project"] for analysis in analyses],
            "total_evaluated_samples": len(working),
            "note": (
                "Project risk levels and concentration are reported separately; "
                "held-out projects are not pooled into one project summary."
            ),
        }
        risk_concentration: dict[str, Any] = {
            "analysis_scope": "per_project",
            "available_in": "projects[].risk_concentration",
        }
        hotspots: list[dict[str, Any]] = []
        signal_profile: dict[str, Any] = {
            "analysis_scope": "per_project",
            "available_in": "projects[].risk_signal_profile",
        }
        temporal: dict[str, Any] = {
            "analysis_scope": "per_project",
            "available_in": "projects[].temporal_analysis",
        }
        developer_priority: dict[str, Any] = {
            "analysis_scope": "per_project",
            "queue_size": sum(len(queue) for queue in queues),
            "rank_semantics": "rank restarts at 1 within each project",
            "priority_score_formula": _PRIORITY_FORMULA,
            "priority_score_is_probability": False,
        }
        boundary_mode = "strict_per_project"
    else:
        project = projects[0] if projects else None
        analysis, queue = _scope_analysis(
            working.drop(columns="_project"),
            explanation_working.drop(columns="_project"),
            project=project,
            risk_threshold=risk_threshold,
        )
        analyses.append(analysis)
        queues.append(queue)
        summary = {"analysis_scope": "single_project", **analysis["summary"]}
        risk_concentration = analysis["risk_concentration"]
        hotspots = analysis["hotspots"]
        signal_profile = analysis["risk_signal_profile"]
        temporal = analysis["temporal_analysis"]
        developer_priority = analysis["developer_priority"]
        boundary_mode = "single_project"

    queue = (
        pd.concat(queues, ignore_index=True)
        if queues
        else pd.DataFrame(columns=DEVELOPER_PRIORITY_COLUMNS)
    )
    artifact = {
        "schema_version": PROJECT_INTELLIGENCE_SCHEMA_VERSION,
        "summary": summary,
        "risk_concentration": risk_concentration,
        "hotspots": hotspots,
        "risk_signal_profile": signal_profile,
        "temporal_analysis": temporal,
        "developer_priority": developer_priority,
        "projects": analyses if preserve_boundaries else [],
        "metadata": _metadata(
            risk_threshold=risk_threshold,
            analyses=analyses,
            unsupported_features=_unsupported_features(
                explanations, actionable_insights
            ),
            boundary_mode=boundary_mode,
        ),
    }
    return ProjectIntelligenceResult(
        artifact=artifact,
        developer_priority=queue.loc[:, DEVELOPER_PRIORITY_COLUMNS],
    )
