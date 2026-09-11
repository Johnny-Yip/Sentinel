"""Developer-facing decision briefs derived from existing V6 outputs.

This module is intentionally downstream of scoring and explanation. It assigns
inspection tiers, selects conservative actions, and summarizes each project
without fitting a model or recomputing any attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil
from typing import Any, Mapping

import numpy as np
import pandas as pd

from sentinel.evaluation.insights import FEATURE_INSIGHTS


DECISION_BRIEF_SCHEMA_VERSION = 1
DEFAULT_MAX_DEVELOPER_ACTIONS = 5
RISK_TIERS = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
ATTENTION_LEVELS = ("LOW", "MODERATE", "ELEVATED", "HIGH")
DEVELOPER_ACTION_COLUMNS = (
    "project",
    "priority_rank",
    "risk_tier",
    "identifier",
    "file_path",
    "snapshot_date",
    "predicted_risk",
    "priority_score",
    "short_reason",
    "dominant_signal",
    "recommended_action",
    "supporting_evidence",
)

_CRITICAL_SHARE = 0.10
_HIGH_CUMULATIVE_SHARE = 0.30
_MODERATE_ATTENTION_MAX_RATE = 0.10
_ELEVATED_ATTENTION_MAX_RATE = 0.25
_LIMITATIONS = [
    "Risk tiers are relative inspection priorities within one project, not "
    "calibrated defect probabilities.",
    "Attention levels summarize the share of samples at the configured risk "
    "cutoff; they are workload heuristics, not probabilities or severity claims.",
    "Recommended actions are review and inspection suggestions. Sentinel does "
    "not claim that an evaluated file contains a defect.",
    "Local explanations describe fitted-model behavior and are not causal "
    "evidence.",
    "Cross-project briefs are independent; absolute ranks and tiers must not be "
    "compared across held-out projects.",
]


@dataclass(frozen=True)
class DecisionBriefResult:
    """Machine-readable brief plus the complete high-risk action queue."""

    artifact: dict[str, Any]
    developer_actions: pd.DataFrame


def _optional_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _optional_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _project_series(frame: pd.DataFrame) -> pd.Series:
    if "project" not in frame:
        return pd.Series([""] * len(frame), index=frame.index, dtype="object")
    return frame["project"].map(_optional_text)


def _series_or_default(
    frame: pd.DataFrame, column: str, default: Any
) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)


def _validate_risk_threshold(risk_threshold: float) -> None:
    if not np.isfinite(risk_threshold) or not 0.0 <= risk_threshold <= 1.0:
        raise ValueError("risk_threshold must be between 0 and 1.")


def attention_level(high_risk_count: int, total_samples: int) -> str:
    """Assign a deterministic inspection-workload attention level.

    LOW means no samples meet the configured risk cutoff. Otherwise MODERATE
    covers rates below 10%, ELEVATED covers rates from 10% up to (but not
    including) 25%, and HIGH covers rates of at least 25%.
    """
    if total_samples < 0 or high_risk_count < 0:
        raise ValueError("Sample counts must be non-negative.")
    if high_risk_count > total_samples:
        raise ValueError("high_risk_count cannot exceed total_samples.")
    if high_risk_count == 0:
        return "LOW"
    rate = high_risk_count / total_samples
    if rate < _MODERATE_ATTENTION_MAX_RATE:
        return "MODERATE"
    if rate < _ELEVATED_ATTENTION_MAX_RATE:
        return "ELEVATED"
    return "HIGH"


def _priority_source(
    ranking: pd.DataFrame, developer_priority: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    if not developer_priority.empty:
        source = developer_priority.copy()
        source_name = "developer_priority"
    else:
        source = pd.DataFrame(
            {
                "project": _project_series(ranking),
                "identifier": (
                    ranking["file_path"].map(_optional_text)
                    if "file_path" in ranking
                    else [""] * len(ranking)
                ),
                "snapshot_date": (
                    ranking["snapshot_date"].map(_optional_text)
                    if "snapshot_date" in ranking
                    else [""] * len(ranking)
                ),
                "predicted_risk": pd.to_numeric(
                    ranking.get("risk_score", pd.Series(dtype=float)),
                    errors="coerce",
                ),
                "priority_score": pd.to_numeric(
                    ranking.get("risk_score", pd.Series(dtype=float)),
                    errors="coerce",
                ),
                "primary_risk_reason": (
                    ranking["primary_risk_reason"]
                    if "primary_risk_reason" in ranking
                    else [None] * len(ranking)
                ),
                "recommended_action": (
                    ranking["recommended_action"]
                    if "recommended_action" in ranking
                    else [None] * len(ranking)
                ),
                "risk_signal_count": pd.to_numeric(
                    _series_or_default(ranking, "risk_signal_count", 0),
                    errors="coerce",
                ).fillna(0),
                "recurring_high_risk_count": [0] * len(ranking),
                "recurring_high_risk_rate": [0.0] * len(ranking),
            }
        )
        source_name = "risk_ranking_fallback"
    required = {"predicted_risk", "priority_score"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(
            "Developer priorities require columns: " + ", ".join(missing)
        )
    for column in (
        "project",
        "identifier",
        "snapshot_date",
        "primary_risk_reason",
        "recommended_action",
    ):
        if column not in source:
            source[column] = ""
        source[column] = source[column].map(_optional_text)
    for column, default in (
        ("risk_signal_count", 0),
        ("recurring_high_risk_count", 0),
        ("recurring_high_risk_rate", 0.0),
    ):
        if column not in source:
            source[column] = default
        source[column] = pd.to_numeric(source[column], errors="coerce").fillna(
            default
        )
    source["predicted_risk"] = pd.to_numeric(
        source["predicted_risk"], errors="coerce"
    )
    source["priority_score"] = pd.to_numeric(
        source["priority_score"], errors="coerce"
    )
    values = source[["predicted_risk", "priority_score"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Decision brief risk and priority scores must be finite.")
    if ((source["predicted_risk"] < 0.0) | (source["predicted_risk"] > 1.0)).any():
        raise ValueError("Decision brief risk scores must be within [0, 1].")
    source["project"] = _project_series(source)
    return source, source_name


def _canonical_priority_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove input-row order from all brief ordering decisions."""
    return frame.sort_values(
        [
            "priority_score",
            "predicted_risk",
            "recurring_high_risk_count",
            "recurring_high_risk_rate",
            "identifier",
            "snapshot_date",
            "primary_risk_reason",
            "recommended_action",
            "risk_signal_count",
        ],
        ascending=[False, False, False, False, True, True, True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def assign_risk_tiers(
    priority: pd.DataFrame, *, risk_threshold: float
) -> pd.DataFrame:
    """Assign project-local tiers without treating scores as probabilities.

    Samples below the existing cutoff are LOW. Within the remaining project-local
    queue, the first ``max(1, ceil(10% * H))`` samples are CRITICAL, samples
    through ``ceil(30% * H)`` are HIGH, and the remainder are MEDIUM, where H is
    the number of samples at or above the cutoff. Ties use canonical content
    fields and never the input row order.
    """
    _validate_risk_threshold(risk_threshold)
    if priority.empty:
        result = priority.copy()
        result["priority_rank"] = pd.Series(dtype="Int64")
        result["risk_tier"] = pd.Series(dtype="object")
        return result
    source = priority.copy()
    source["project"] = _project_series(source)
    tiered: list[pd.DataFrame] = []
    for _, project_rows in source.groupby("project", sort=True, dropna=False):
        ordered = _canonical_priority_order(project_rows)
        eligible = ordered["predicted_risk"] >= risk_threshold
        high_risk_count = int(eligible.sum())
        critical_end = (
            max(1, ceil(_CRITICAL_SHARE * high_risk_count))
            if high_risk_count
            else 0
        )
        high_end = max(
            critical_end, ceil(_HIGH_CUMULATIVE_SHARE * high_risk_count)
        )
        priority_ranks: list[int | None] = []
        tiers: list[str] = []
        current_high_rank = 0
        for is_eligible in eligible.tolist():
            if not is_eligible:
                priority_ranks.append(None)
                tiers.append("LOW")
                continue
            current_high_rank += 1
            priority_ranks.append(current_high_rank)
            if current_high_rank <= critical_end:
                tiers.append("CRITICAL")
            elif current_high_rank <= high_end:
                tiers.append("HIGH")
            else:
                tiers.append("MEDIUM")
        ordered["priority_rank"] = pd.array(priority_ranks, dtype="Int64")
        ordered["risk_tier"] = tiers
        tiered.append(ordered)
    return pd.concat(tiered, ignore_index=True)


def _analysis_projects(project_intelligence: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary = project_intelligence.get("summary", {})
    if summary.get("analysis_scope") == "per_project":
        return list(project_intelligence.get("projects", []))
    if summary:
        return [
            {
                "project": summary.get("project"),
                "summary": summary,
                "risk_concentration": project_intelligence.get(
                    "risk_concentration", {}
                ),
                "hotspots": project_intelligence.get("hotspots", []),
                "risk_signal_profile": project_intelligence.get(
                    "risk_signal_profile", {}
                ),
                "temporal_analysis": project_intelligence.get(
                    "temporal_analysis", {}
                ),
                "developer_priority": project_intelligence.get(
                    "developer_priority", {}
                ),
            }
        ]
    return []


def _explanation_index(explanations: pd.DataFrame) -> dict[tuple[str, str, str], pd.DataFrame]:
    required = {"project", "file_path", "snapshot_date"}
    if explanations.empty or not required <= set(explanations.columns):
        return {}
    working = explanations.copy()
    for column in required:
        working[column] = working[column].map(_optional_text)
    index: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, group in working.groupby(
        ["project", "file_path", "snapshot_date"], sort=True, dropna=False
    ):
        index[tuple(str(value) for value in key)] = group.copy()
    return index


def _actionable_index(
    actionable_insights: Mapping[str, Any],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    samples = actionable_insights.get("samples", [])
    for sample in sorted(
        samples,
        key=lambda row: (
            _optional_text(row.get("project")),
            _optional_text(row.get("file_path")),
            _optional_text(row.get("snapshot_date")),
            _optional_text(row.get("sample_id")),
        ),
    ):
        key = (
            _optional_text(sample.get("project")),
            _optional_text(sample.get("file_path")),
            _optional_text(sample.get("snapshot_date")),
        )
        indexed.setdefault(key, dict(sample))
    return indexed


def _dominant_explanation(
    explanations: pd.DataFrame,
) -> dict[str, Any] | None:
    if explanations.empty or not {
        "feature",
        "contribution",
        "direction",
    } <= set(explanations.columns):
        return None
    working = explanations.loc[
        explanations["direction"] == "increases_risk"
    ].copy()
    working = working.loc[working["feature"].astype(str).isin(FEATURE_INSIGHTS)]
    if working.empty:
        return None
    working["_absolute"] = pd.to_numeric(
        working.get("absolute_contribution", working["contribution"].abs()),
        errors="coerce",
    ).fillna(0.0)
    ordered = working.sort_values(
        ["_absolute", "feature", "contribution"],
        ascending=[False, True, False],
        kind="stable",
    )
    return ordered.iloc[0].to_dict()


def _supporting_evidence(
    row: Any,
    dominant: Mapping[str, Any] | None,
    *,
    risk_threshold: float,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "risk_score": float(row.predicted_risk),
        "risk_threshold": risk_threshold,
        "recurring_high_risk_count": int(row.recurring_high_risk_count),
        "recurring_high_risk_rate": float(row.recurring_high_risk_rate),
        "risk_signal_count": int(row.risk_signal_count),
    }
    if dominant is not None:
        evidence["dominant_signal"] = str(dominant["feature"])
        for source, destination in (
            ("feature_value", "feature_value"),
            ("contribution", "contribution"),
            ("normalized_contribution", "normalized_contribution"),
        ):
            value = dominant.get(source)
            number = _optional_number(value)
            if number is not None:
                evidence[destination] = number
        for source in ("explanation_method", "contribution_space"):
            value = _optional_text(dominant.get(source))
            if value:
                evidence[source] = value
    return evidence


def _project_actions(
    tiered: pd.DataFrame,
    explanations: dict[tuple[str, str, str], pd.DataFrame],
    actionable: dict[tuple[str, str, str], dict[str, Any]],
    *,
    project: str,
    risk_threshold: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    selected = tiered.loc[
        (tiered["project"] == project) & tiered["priority_rank"].notna()
    ].sort_values("priority_rank", kind="stable")
    artifact_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        key = (project, row.identifier, row.snapshot_date)
        identifiers_complete = bool(row.identifier and row.snapshot_date)
        actionable_sample = actionable.get(key, {}) if identifiers_complete else {}
        dominant = (
            _dominant_explanation(explanations.get(key, pd.DataFrame()))
            if identifiers_complete
            else None
        )
        identifier = row.identifier or None
        reason = _optional_text(
            actionable_sample.get("primary_risk_reason")
        ) or _optional_text(row.primary_risk_reason)
        recommendation = _optional_text(
            actionable_sample.get("recommended_action")
        ) or _optional_text(row.recommended_action)
        evidence = _supporting_evidence(
            row, dominant, risk_threshold=risk_threshold
        )
        action = {
            "project": project or None,
            "priority_rank": int(row.priority_rank),
            "risk_tier": str(row.risk_tier),
            "identifier": identifier,
            "file_path": identifier,
            "snapshot_date": row.snapshot_date or None,
            "predicted_risk": float(row.predicted_risk),
            "priority_score": float(row.priority_score),
            "short_reason": reason or None,
            "dominant_signal": (
                str(dominant["feature"]) if dominant is not None else None
            ),
            "recommended_action": recommendation or None,
            "supporting_evidence": evidence,
        }
        artifact_rows.append(action)
        csv_rows.append(
            {
                **action,
                "supporting_evidence": json.dumps(
                    evidence, sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return artifact_rows, pd.DataFrame(csv_rows, columns=DEVELOPER_ACTION_COLUMNS)


def _highest_risk_signal(analysis: Mapping[str, Any]) -> dict[str, Any] | None:
    profile = analysis.get("risk_signal_profile", {})
    records = profile.get("strongest_aggregate_risk_contributors", [])
    if not records:
        return None
    first = records[0]
    return {
        key: first.get(key)
        for key in (
            "feature",
            "category",
            "occurrence_count",
            "aggregate_contribution",
            "contribution_spaces",
            "explanation_methods",
        )
    }


def _top_hotspot(analysis: Mapping[str, Any]) -> dict[str, Any] | None:
    hotspots = analysis.get("hotspots", [])
    if not hotspots:
        return None
    first = hotspots[0]
    dominant = first.get("dominant_risk_signals", [])
    return {
        "identifier": first.get("identifier"),
        "identifier_type": first.get("identifier_type"),
        "high_risk_count": first.get("high_risk_count"),
        "high_risk_rate": first.get("high_risk_rate"),
        "mean_risk": first.get("mean_risk"),
        "dominant_signal": dominant[0].get("feature") if dominant else None,
    }


def _concentration_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    concentration = analysis.get("risk_concentration", {})
    top_10 = concentration.get(
        "top_10_percent_risk_contribution_percentage", 0.0
    )
    top_20 = concentration.get(
        "top_20_percent_risk_contribution_percentage", 0.0
    )
    samples_for_half = concentration.get(
        "samples_responsible_for_50_percent_of_predicted_risk", 0
    )
    return {
        "top_10_percent_risk_share_percentage": float(top_10),
        "top_20_percent_risk_share_percentage": float(top_20),
        "samples_for_50_percent_of_risk": int(samples_for_half),
        "summary": (
            f"The top 10% of samples account for {float(top_10):.2f}% of "
            f"aggregate predicted risk; {int(samples_for_half)} sample(s) "
            "account for its first 50%."
        ),
    }


def build_decision_brief(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    actionable_insights: Mapping[str, Any],
    project_intelligence: Mapping[str, Any],
    developer_priority: pd.DataFrame,
    *,
    risk_threshold: float,
    experiment_type: str,
    model_metadata: Mapping[str, Any] | None = None,
    max_actions_per_project: int = DEFAULT_MAX_DEVELOPER_ACTIONS,
) -> DecisionBriefResult:
    """Build one independent developer decision brief per evaluated project."""
    _validate_risk_threshold(risk_threshold)
    if max_actions_per_project <= 0:
        raise ValueError("max_actions_per_project must be positive.")
    priority, priority_source = _priority_source(ranking, developer_priority)
    tiered = assign_risk_tiers(priority, risk_threshold=risk_threshold)
    explanation_index = _explanation_index(explanations)
    actionable_index = _actionable_index(actionable_insights)
    analyses = sorted(
        _analysis_projects(project_intelligence),
        key=lambda row: _optional_text(row.get("project")),
    )
    projects: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    ranking_working = ranking.copy()
    ranking_working["_project"] = _project_series(ranking_working)
    for analysis in analyses:
        project = _optional_text(analysis.get("project"))
        actions, action_frame = _project_actions(
            tiered,
            explanation_index,
            actionable_index,
            project=project,
            risk_threshold=risk_threshold,
        )
        action_frames.append(action_frame)
        summary = analysis.get("summary", {})
        total_samples = int(summary.get("total_evaluated_samples", 0))
        high_risk_count = int(summary.get("high_risk_sample_count", len(actions)))
        high_risk_rate = high_risk_count / total_samples if total_samples else 0.0
        tier_counts = {tier: 0 for tier in RISK_TIERS}
        project_tiers = tiered.loc[tiered["project"] == project, "risk_tier"]
        for tier, count in project_tiers.value_counts().items():
            tier_counts[str(tier)] = int(count)
        temporal = analysis.get("temporal_analysis", {})
        projects.append(
            {
                "project": project or None,
                "executive_summary": {
                    "total_samples": total_samples,
                    "high_risk_count": high_risk_count,
                    "high_risk_rate": high_risk_rate,
                    "mean_risk": summary.get("mean_predicted_risk"),
                    "concentration_summary": _concentration_summary(analysis),
                    "top_recurring_hotspot": _top_hotspot(analysis),
                    "temporal_direction": temporal.get(
                        "direction", "not_available"
                    ),
                    "highest_risk_signal": _highest_risk_signal(analysis),
                    "number_of_priority_items": len(actions),
                    "overall_attention_level": attention_level(
                        high_risk_count, total_samples
                    ),
                    "risk_tier_counts": tier_counts,
                },
                "top_developer_actions": actions[:max_actions_per_project],
                "models_used": sorted(
                    ranking_working.loc[ranking_working["_project"] == project]
                    .get("model", pd.Series(dtype=str))
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                ),
            }
        )
    actions_frame = (
        pd.concat(action_frames, ignore_index=True)
        if action_frames
        else pd.DataFrame(columns=DEVELOPER_ACTION_COLUMNS)
    )
    artifact = {
        "schema_version": DECISION_BRIEF_SCHEMA_VERSION,
        "experiment_type": experiment_type,
        "analysis_scope": (
            "per_project"
            if experiment_type == "cross_project" or len(projects) > 1
            else "single_project"
        ),
        "project_count": len(projects),
        "projects": projects,
        "metadata": {
            "deterministic": True,
            "input_dependencies": [
                "risk_ranking",
                "prediction_explanations",
                "actionable_insights",
                "project_intelligence",
                priority_source,
            ],
            "risk_threshold": risk_threshold,
            "risk_threshold_semantics": (
                "Existing evaluation cutoff used to identify the project-local "
                "inspection candidate set; it is not a calibrated probability."
            ),
            "risk_tier_definitions": {
                "CRITICAL": (
                    "Top max(1, ceil(10% * H)) project-local priority items among "
                    "the H samples at or above the configured risk cutoff."
                ),
                "HIGH": (
                    "Remaining project-local priority items through ceil(30% * H)."
                ),
                "MEDIUM": (
                    "Remaining project-local samples at or above the configured "
                    "risk cutoff."
                ),
                "LOW": "Samples below the configured risk cutoff.",
            },
            "attention_level_definitions": {
                "LOW": "No samples are at or above the configured risk cutoff.",
                "MODERATE": "A non-zero share below 10% meets the cutoff.",
                "ELEVATED": "At least 10% but less than 25% meets the cutoff.",
                "HIGH": "At least 25% meets the cutoff.",
            },
            "tie_breaking_rules": (
                "priority_score_desc, predicted_risk_desc, "
                "recurring_high_risk_count_desc, recurring_high_risk_rate_desc, "
                "identifier_asc, snapshot_date_asc, short_reason_asc, "
                "recommended_action_asc, risk_signal_count_desc"
            ),
            "recommended_action_semantics": (
                "Conservative inspection suggestions copied from the existing "
                "actionable insight registry; they do not assert a defect."
            ),
            "feature_registry": {
                "source": "sentinel.evaluation.insights.FEATURE_INSIGHTS",
                "supported_features": sorted(FEATURE_INSIGHTS),
            },
            "model_metadata": dict(model_metadata or {}),
            "max_actions_per_project": max_actions_per_project,
            "limitations": list(_LIMITATIONS),
        },
    }
    return DecisionBriefResult(artifact=artifact, developer_actions=actions_frame)


def _display(value: Any, *, percentage: bool = False) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):.2%}" if percentage else f"{float(value):.4f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    escaped = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in escaped),
    ]


def decision_brief_markdown_lines(
    artifact: Mapping[str, Any], *, embedded: bool = False
) -> list[str]:
    """Render concise standalone or report-embedded Markdown lines."""
    project_heading = "###" if embedded else "##"
    lines: list[str] = []
    if artifact.get("analysis_scope") == "per_project":
        lines.extend(
            [
                "Each held-out project is summarized independently; ranks and "
                "tiers are not pooled or compared across projects.",
                "",
            ]
        )
    for project in artifact.get("projects", []):
        summary = project["executive_summary"]
        project_name = project.get("project") or "project unavailable"
        hotspot = summary.get("top_recurring_hotspot")
        signal = summary.get("highest_risk_signal")
        lines.extend(
            [
                f"{project_heading} {project_name}",
                "",
                f"- Overall attention: **{summary['overall_attention_level']}** "
                "(inspection heuristic).",
                f"- High-risk concentration: {summary['high_risk_count']:,} of "
                f"{summary['total_samples']:,} samples "
                f"({_display(summary['high_risk_rate'], percentage=True)}) meet "
                "the configured cutoff; "
                f"{summary['concentration_summary']['summary']}",
                "- Top recurring hotspot: "
                + (
                    f"`{hotspot['identifier']}` "
                    f"({hotspot['high_risk_count']} high-risk observations)."
                    if hotspot and hotspot.get("identifier")
                    else "not available."
                ),
                f"- Temporal direction: {summary['temporal_direction']}.",
                "- Highest-risk signal: "
                + (
                    f"`{signal['feature']}`."
                    if signal and signal.get("feature")
                    else "not available."
                ),
                "",
                "Top developer actions:",
                "",
            ]
        )
        actions = project.get("top_developer_actions", [])[:5]
        if actions:
            lines.extend(
                _markdown_table(
                    [
                        "Rank",
                        "Tier",
                        "Identifier",
                        "Risk",
                        "Reason",
                        "Review suggestion",
                    ],
                    [
                        [
                            str(action["priority_rank"]),
                            str(action["risk_tier"]),
                            str(action.get("identifier") or "unavailable"),
                            _display(action["predicted_risk"]),
                            str(action.get("short_reason") or "not available"),
                            str(action.get("recommended_action") or "not available"),
                        ]
                        for action in actions
                    ],
                )
            )
        else:
            lines.append("No samples met the configured risk cutoff.")
        lines.append("")
    lines.extend(
        [
            "Interpretation warning: tiers and attention levels are deterministic "
            "inspection heuristics, not calibrated defect probabilities. Actions "
            "suggest review; they do not assert that a defect exists.",
        ]
    )
    return lines


def build_decision_brief_markdown(artifact: Mapping[str, Any]) -> str:
    """Build the standalone ``developer_risk_brief.md`` artifact."""
    return "\n".join(
        [
            "# Sentinel Developer Risk Brief",
            "",
            *decision_brief_markdown_lines(artifact),
            "",
        ]
    )
