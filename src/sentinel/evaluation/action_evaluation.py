"""Diagnostic evaluation of stored explanations and developer actions.

This module consumes artifacts only. It never fits, scores, explains, selects
models, or changes thresholds. Missing evidence is not inferred from risk.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


ACTION_EVALUATION_SCHEMA_VERSION = 1
STABILITY_MAX_GAP_DAYS = 62
_FLAGS = (
    "has_identifier",
    "has_timestamp",
    "has_hotspot_evidence",
    "has_explanation_evidence",
    "has_signal_evidence",
)
ACTION_QUALITY_COLUMNS = (
    "project",
    "rank",
    "tier",
    "identifier",
    "snapshot_date",
    "risk_score",
    "evidence_coverage_score",
    "specificity_score",
    "specificity_label",
    *_FLAGS,
    "evidence_item_count",
    "explanation_item_count",
    "signal_item_count",
    "references_identifier",
    "references_signal_or_feature",
    "has_concrete_recommendation",
    "avoids_generic_guidance",
    "recurring_high_risk_count",
    "risk_signal_count",
    "dominant_signal",
    "short_reason",
    "recommended_action",
    "supporting_explanations",
    "supporting_signals",
    "supporting_evidence",
)
_CONTEXT = ("model", "evaluation_mode", "explanation_method", "contribution_space")
_LIMITATIONS = [
    "These scores are diagnostic evaluation heuristics, not proof of explanation "
    "correctness, causal correctness, human usefulness, or defect prevention.",
    "Stability compares nearby observed snapshots of the same file, not verified "
    "semantic similarity. Missing, ambiguous, zero, or incompatible vectors "
    "cannot supply cosine comparisons.",
    "Vector diagnostics use only retained prediction_explanations rows. Top-k "
    "truncation can inflate top-feature share and omit explanation magnitude; "
    "unobserved features are never filled with zero.",
    "Strength is the sum of absolute retained contributions, not a probability. "
    "A project mean is unavailable across mixed or unknown models, explanation "
    "methods, or contribution spaces.",
    "Coverage flags indicate observed evidence availability. False flags and zero "
    "item counts do not prove that real-world evidence does not exist. A recurring "
    "hotspot is not expected for every useful action.",
    "Specificity checks structured action references and a fixed English inspection "
    "vocabulary; it does not evaluate recommendation quality or effectiveness.",
    "Each project is evaluated independently. These diagnostics never select "
    "models or thresholds, and should not be compared across held-out models.",
]


@dataclass(frozen=True)
class ActionEvaluationResult:
    """Project diagnostics and one quality row for each existing action."""

    artifact: dict[str, Any]
    action_quality: pd.DataFrame


def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return (
        None
        if text.casefold()
        in {
            "",
            "nan",
            "nat",
            "none",
            "null",
            "<na>",
            "unknown",
            "unavailable",
            "not_available",
        }
        else text
    )


def _number(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> pd.Timestamp | None:
    text = _text(value)
    if text is None:
        return None
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    return None if pd.isna(parsed) else parsed


def _key(row: Mapping[str, Any], path_column: str = "file_path") -> tuple | None:
    project = _text(row.get("project"))
    path = _text(row.get(path_column))
    date = _timestamp(row.get("snapshot_date"))
    return (project, path, date) if project and path and date is not None else None


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is None or pd.isna(value):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return {}
    return _clean(value) if isinstance(value, Mapping) else {}


def _json(value: Any) -> str | None:
    return (
        json.dumps(
            _clean(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if value
        else None
    )


def _mean(values: list[float]) -> float | None:
    return (
        _number(math.fsum(value / len(values) for value in values)) if values else None
    )


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else ordered[mid - 1] / 2 + ordered[mid] / 2


@dataclass
class _Vector:
    key: tuple
    context: tuple
    weights: dict[str, float]
    complete: bool

    @property
    def nonzero(self) -> bool:
        return any(self.weights.values())

    def magnitude(self) -> tuple[float | None, float | None]:
        if not self.complete or not self.weights:
            return None, None
        maximum = max(abs(value) for value in self.weights.values())
        if not maximum:
            return 0.0, None
        scaled_sum = math.fsum(abs(value) / maximum for value in self.weights.values())
        return _number(maximum * scaled_sum), 1.0 / scaled_sum


def _matched_vectors(
    ranking: list[dict[str, Any]],
    explanations: pd.DataFrame,
) -> list[_Vector | None]:
    """Join only unique evaluated identities; never attach an orphan explanation."""
    indexed: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in explanations.to_dict("records"):
        key = _key(row)
        if key is not None:
            indexed[key].append(row)
    identities = [
        (_key(row), _text(row.get("model")), _text(row.get("evaluation_mode")))
        for row in ranking
    ]
    counts = Counter(identities)
    result: list[_Vector | None] = []
    for identity in identities:
        key, model, mode = identity
        candidates = indexed.get(key, [])
        if key is None or counts[identity] != 1:
            result.append(None)
            continue
        candidates = [
            record
            for record in candidates
            if (model is None or _text(record.get("model")) == model)
            and (mode is None or _text(record.get("evaluation_mode")) == mode)
        ]
        contexts = {
            tuple(_text(record.get(column)) for column in _CONTEXT)
            for record in candidates
        }
        features = [_text(record.get("feature")) for record in candidates]
        if not candidates or len(contexts) != 1 or len(set(features)) != len(features):
            result.append(None)
            continue
        weights = {
            feature: number
            for feature, record in zip(features, candidates)
            if feature is not None
            and (number := _number(record.get("contribution"))) is not None
        }
        result.append(
            _Vector(key, next(iter(contexts)), weights, len(weights) == len(candidates))
        )
    return result


def _stability(
    ranking: list[dict[str, Any]],
    vectors: list[_Vector | None],
) -> dict[str, Any]:
    # Adjacency is determined before removing zero/incomplete vectors; an
    # unavailable middle observation must not create a new neighboring pair.
    groups: dict[tuple, list[tuple[pd.Timestamp, _Vector | None]]] = defaultdict(list)
    for row, vector in zip(ranking, vectors):
        key = _key(row)
        if key is not None:
            groups[key[:2]].append((key[2], vector))
    similarities: list[float] = []
    used: set[tuple] = set()
    for group in groups.values():
        ordered = sorted(group, key=lambda item: item[0])
        date_counts = Counter(date for date, _ in ordered)
        for (left_date, left), (right_date, right) in zip(ordered, ordered[1:]):
            if (
                left is None
                or right is None
                or date_counts[left_date] != 1
                or date_counts[right_date] != 1
            ):
                continue
            gap = (right_date - left_date).total_seconds() / 86400
            if (
                not 0 < gap <= STABILITY_MAX_GAP_DAYS
                or left.context != right.context
                or not all(left.context)
                or not left.complete
                or not right.complete
                or not left.nonzero
                or not right.nonzero
                or set(left.weights) != set(right.weights)
            ):
                continue
            features = sorted(left.weights)
            a = np.array([left.weights[name] for name in features])
            b = np.array([right.weights[name] for name in features])
            a = a / np.max(np.abs(a))
            b = b / np.max(np.abs(b))
            similarity = float(
                np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1.0, 1.0)
            )
            similarities.append(similarity)
            used.update((left.key, right.key))
    return {
        "mean_explanation_similarity": _mean(similarities),
        "median_explanation_similarity": _median(similarities),
        "stability_sample_count": len(used),
        "stability_pair_count": len(similarities),
    }


_INSPECTION_VERB = re.compile(
    r"\b(inspect|review|check|audit|trace|verify|examine|test|compare|confirm)\b", re.I
)
_INSPECTION_TARGET = re.compile(
    r"\b(edits?|changes?|diffs?|rewrites?|rework|fixes|regressions?|tests?|coverage|"
    r"ownership|owners?|maintainers?|handoffs?|commits?|churn|hotspots?|dependencies|"
    r"interfaces?|callers?|boundaries|failures?|history|coupling)\b",
    re.I,
)
_GENERIC = re.compile(
    r"\b(review (?:the )?(?:code|detailed (?:prediction|explanation))|"
    r"improve (?:code )?quality|follow best practices|be careful)\b",
    re.I,
)


def _quality_row(action: Mapping[str, Any], vector: _Vector | None) -> dict[str, Any]:
    identifier = _text(action.get("identifier")) or _text(action.get("file_path"))
    timestamp = (
        _text(action.get("snapshot_date"))
        if _timestamp(action.get("snapshot_date")) is not None
        else None
    )
    evidence = _evidence(action.get("supporting_evidence"))
    dominant = _text(action.get("dominant_signal"))
    named_signals = sorted(
        {name for name in (dominant, _text(evidence.get("dominant_signal"))) if name}
    )
    explanations = {
        feature: {
            "feature": feature,
            "contribution": contribution,
            "explanation_method": vector.context[2],
            "contribution_space": vector.context[3],
            "source": "prediction_explanations",
        }
        for feature, contribution in (vector.weights.items() if vector else [])
        if contribution != 0.0
    }
    embedded_feature = _text(evidence.get("dominant_signal")) or dominant
    embedded_weight = _number(evidence.get("contribution"))
    if embedded_feature and embedded_weight is not None and embedded_weight != 0:
        explanations.setdefault(
            embedded_feature,
            {
                "feature": embedded_feature,
                "contribution": embedded_weight,
                "explanation_method": _text(evidence.get("explanation_method")),
                "contribution_space": _text(evidence.get("contribution_space")),
                "source": "developer_actions.supporting_evidence",
            },
        )
    signal_count = _number(evidence.get("risk_signal_count"))
    if signal_count is not None and (signal_count < 0 or not signal_count.is_integer()):
        signal_count = None
    hotspot_count = _number(evidence.get("recurring_high_risk_count"))
    if hotspot_count is not None and (
        hotspot_count < 0 or not hotspot_count.is_integer()
    ):
        hotspot_count = None
    signal_items = max(len(named_signals), int(signal_count or 0))
    flags = {
        "has_identifier": identifier is not None,
        "has_timestamp": timestamp is not None,
        "has_hotspot_evidence": bool(
            identifier and hotspot_count is not None and hotspot_count >= 2
        ),
        "has_explanation_evidence": bool(explanations),
        "has_signal_evidence": signal_items > 0,
    }
    recommendation = _text(action.get("recommended_action"))
    recommendation_text = recommendation or ""
    guidance_text = (
        recommendation_text + " " + (_text(action.get("short_reason")) or "")
    )
    references = bool(
        named_signals
        or any(
            feature.casefold() in guidance_text.casefold() for feature in explanations
        )
    )
    explicit_target = any(
        name.casefold() in recommendation_text.casefold()
        for name in [identifier, *named_signals, *explanations]
        if name
    )
    concrete = bool(
        _INSPECTION_VERB.search(recommendation_text)
        and (_INSPECTION_TARGET.search(recommendation_text) or explicit_target)
    )
    generic = bool(_GENERIC.search(recommendation_text)) and not explicit_target
    concrete = concrete and not generic
    specificity_flags = {
        "references_identifier": identifier is not None,
        "references_signal_or_feature": references,
        "has_concrete_recommendation": concrete,
        "avoids_generic_guidance": concrete and bool(identifier or references),
    }
    score = sum(specificity_flags.values()) / 4.0
    return {
        "project": _text(action.get("project")),
        "rank": _number(action.get("priority_rank")),
        "tier": _text(action.get("risk_tier")),
        "identifier": identifier,
        "snapshot_date": timestamp,
        "risk_score": _number(action.get("predicted_risk")),
        "evidence_coverage_score": sum(flags.values()) / 5.0,
        "specificity_score": score,
        "specificity_label": (
            "HIGH" if score >= 0.75 else "MODERATE" if score >= 0.5 else "LOW"
        ),
        **flags,
        "evidence_item_count": int(flags["has_identifier"])
        + int(flags["has_timestamp"])
        + int(flags["has_hotspot_evidence"])
        + len(explanations)
        + signal_items,
        "explanation_item_count": len(explanations),
        "signal_item_count": signal_items,
        **specificity_flags,
        "recurring_high_risk_count": hotspot_count,
        "risk_signal_count": signal_count,
        "dominant_signal": dominant,
        "short_reason": _text(action.get("short_reason")),
        "recommended_action": recommendation,
        "supporting_explanations": _json(
            [explanations[key] for key in sorted(explanations)]
        ),
        "supporting_signals": _json(named_signals),
        "supporting_evidence": _json(evidence),
    }


def _alignment(
    ranking: list[dict[str, Any]],
    vectors: list[_Vector | None],
    threshold: float,
) -> dict[str, Any]:
    high_risk = [
        (row, vector)
        for row, vector in zip(ranking, vectors)
        if (score := _number(row.get("risk_score"))) is not None and score >= threshold
    ]
    explained = [
        vector for _, vector in high_risk if vector is not None and vector.nonzero
    ]
    strengths: list[float] = []
    shares: list[float] = []
    units: set[tuple] = set()
    for vector in explained:
        strength, share = vector.magnitude()
        units.add(vector.context)
        if strength is not None:
            strengths.append(strength)
        if share is not None:
            shares.append(share)
    compatible = len(units) == 1 and all(next(iter(units)))
    return {
        "high_risk_sample_count": len(high_risk),
        "explained_high_risk_count": len(explained),
        "explanation_coverage_rate": (
            len(explained) / len(high_risk) if high_risk else None
        ),
        "mean_top_feature_share": _mean(shares),
        "mean_explanation_strength": _mean(strengths) if compatible else None,
        "top_feature_share_sample_count": len(shares),
        "explanation_strength_sample_count": len(strengths) if compatible else 0,
    }


def build_action_evaluation(
    ranking: pd.DataFrame,
    explanations: pd.DataFrame,
    developer_actions: pd.DataFrame,
    *,
    risk_threshold: float,
    experiment_type: str,
) -> ActionEvaluationResult:
    """Evaluate existing artifacts independently for every observed project.

    Coverage = available categories / 5 (identifier, timestamp, recurring
    hotspot, nonzero explanation, risk signal). Specificity = satisfied checks
    / 4 (identifier, named signal/feature, concrete inspection, non-generic).
    Unknown identities are never manufactured or used for an evidence join.
    """
    threshold = _number(risk_threshold)
    if threshold is None or not 0 <= threshold <= 1:
        raise ValueError("risk_threshold must be between 0 and 1.")
    if experiment_type not in {"within_project", "cross_project"}:
        raise ValueError("experiment_type must be within_project or cross_project.")
    ranking_rows = ranking.to_dict("records")
    action_rows = developer_actions.to_dict("records")
    project_names = {_text(row.get("project")) for row in ranking_rows + action_rows}
    if not project_names and experiment_type == "within_project":
        project_names.add(None)
    projects: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for project in sorted(project_names, key=lambda value: value or ""):
        samples = [row for row in ranking_rows if _text(row.get("project")) == project]
        actions = [row for row in action_rows if _text(row.get("project")) == project]
        vectors = _matched_vectors(samples, explanations)
        # Actions have no model/mode columns in Phase 5. Only unique ranked
        # identities can supply additional evidence; embedded evidence is kept.
        vector_index: dict[tuple, list[_Vector | None]] = defaultdict(list)
        for sample, vector in zip(samples, vectors):
            key = _key(sample)
            if key is not None:
                vector_index[key].append(vector)
        rows = []
        for action in actions:
            action_key = _key(
                {
                    **action,
                    "identifier": _text(action.get("identifier"))
                    or _text(action.get("file_path")),
                },
                "identifier",
            )
            matches = vector_index.get(action_key, [])
            rows.append(_quality_row(action, matches[0] if len(matches) == 1 else None))
        rows.sort(
            key=lambda row: (
                row["rank"] is None,
                row["rank"] or 0,
                json.dumps(row, sort_keys=True, allow_nan=False),
            )
        )
        quality_rows.extend(rows)
        coverage = [row["evidence_coverage_score"] for row in rows]
        specificity = [row["specificity_score"] for row in rows]
        projects.append(
            {
                "project": project,
                "evaluation_counts": {
                    "evaluated_sample_count": len(samples),
                    "action_count": len(rows),
                    "explained_sample_count": sum(
                        vector is not None and vector.nonzero for vector in vectors
                    ),
                    "samples_with_unknown_risk": sum(
                        _number(row.get("risk_score")) is None for row in samples
                    ),
                },
                "explanation_stability": _stability(samples, vectors),
                "evidence_coverage": {
                    "action_count": len(rows),
                    "mean_evidence_coverage_score": _mean(coverage),
                    "median_evidence_coverage_score": _median(coverage),
                    "evidence_item_count": sum(
                        row["evidence_item_count"] for row in rows
                    ),
                    "flag_counts": {
                        flag: sum(row[flag] for row in rows) for flag in _FLAGS
                    },
                },
                "action_specificity": {
                    "action_count": len(rows),
                    "mean_specificity_score": _mean(specificity),
                    "median_specificity_score": _median(specificity),
                    "specificity_label_counts": {
                        label: sum(row["specificity_label"] == label for row in rows)
                        for label in ("LOW", "MODERATE", "HIGH")
                    },
                },
                "risk_explanation_alignment": _alignment(samples, vectors, threshold),
                "limitations": list(_LIMITATIONS),
            }
        )
    artifact = {
        "schema_version": ACTION_EVALUATION_SCHEMA_VERSION,
        "experiment_type": experiment_type,
        "analysis_scope": (
            "per_project"
            if experiment_type == "cross_project" or len(projects) > 1
            else "single_project"
        ),
        "project_count": len(projects),
        "projects": projects,
        "evaluation_counts": {
            "evaluated_sample_count": len(ranking_rows),
            "action_count": len(action_rows),
        },
        "metadata": {
            "deterministic": True,
            "risk_threshold": threshold,
            "input_dependencies": [
                "risk_ranking",
                "prediction_explanations",
                "developer_actions",
            ],
            "stability_method": "signed_cosine_adjacent_same_file_snapshots",
            "stability_max_gap_days": STABILITY_MAX_GAP_DAYS,
            "stability_comparability": "Same project, path, model, evaluation mode, explanation method, contribution space, and identical retained feature sets; finite nonzero vectors only.",
            "evidence_coverage_formula": "sum(has_identifier, has_timestamp, has_hotspot_evidence, has_explanation_evidence, has_signal_evidence) / 5",
            "evidence_item_count_formula": "identifier + timestamp + hotspot_record + nonzero_explanation_items + signal_items; each first category counts at most once",
            "signal_items": "Maximum of distinct attached signal names and an available nonnegative integer risk_signal_count; names are never invented from a count.",
            "hotspot_evidence": "An identifier plus recurring_high_risk_count >= 2 in attached action evidence.",
            "specificity_formula": "sum(references_identifier, references_signal_or_feature, has_concrete_recommendation, avoids_generic_guidance) / 4",
            "specificity_labels": {
                "LOW": "score < 0.5",
                "MODERATE": "0.5 <= score < 0.75",
                "HIGH": "score >= 0.75",
            },
            "alignment": "risk_score >= configured risk_threshold; explained means at least one matched finite nonzero named contribution; share=max(abs(weights))/sum(abs(weights)); strength=sum(abs(weights))",
            "missing_values": "Undefined means/rates are null; counts describe only observed evidence. Unknown identities cannot be joined. Strength and share means use their reported eligible sample counts.",
        },
        "limitations": list(_LIMITATIONS),
    }
    return ActionEvaluationResult(
        artifact, pd.DataFrame(quality_rows, columns=ACTION_QUALITY_COLUMNS)
    )


def action_evaluation_markdown_lines(
    artifact: Mapping[str, Any],
    *,
    embedded: bool = False,
) -> list[str]:
    """Render project-local diagnostics without implying validated usefulness."""

    def display(value: Any) -> str:
        return "unknown" if value is None else f"{value:.4f}"

    lines = [
        "Diagnostic evaluation heuristics only; these scores do not establish "
        "explanation correctness, causal correctness, human usefulness, or defect prevention.",
        "",
    ]
    if artifact.get("analysis_scope") == "per_project":
        lines.extend(["Each held-out project is evaluated independently.", ""])
    for project in artifact.get("projects", []):
        stability = project["explanation_stability"]
        alignment = project["risk_explanation_alignment"]
        name = (project.get("project") or "project unknown").replace("\n", " ")
        lines.extend(
            [
                f"{'###' if embedded else '##'} {name}",
                "",
                f"- Evaluated samples: {project['evaluation_counts']['evaluated_sample_count']}; actions: {project['evaluation_counts']['action_count']}.",
                f"- Explanation stability: mean cosine {display(stability['mean_explanation_similarity'])}, median {display(stability['median_explanation_similarity'])}; {stability['stability_sample_count']} samples in {stability['stability_pair_count']} comparable pairs.",
                f"- Evidence coverage: mean {display(project['evidence_coverage']['mean_evidence_coverage_score'])} (five equally weighted availability flags).",
                f"- Action specificity: mean {display(project['action_specificity']['mean_specificity_score'])} (four deterministic checks).",
                f"- High-risk explanation coverage: {alignment['explained_high_risk_count']}/{alignment['high_risk_sample_count']}; rate {display(alignment['explanation_coverage_rate'])}.",
                f"- Retained-vector top-feature share: {display(alignment['mean_top_feature_share'])}; strength: {display(alignment['mean_explanation_strength'])}.",
                "",
            ]
        )
    lines.extend(
        [
            "Stability uses neighboring snapshots of the same file within 62 days "
            "and compatible retained feature sets. Missing comparisons remain unknown; "
            "top-k truncation limits vector diagnostics. Strength is unavailable across "
            "incompatible explanation units.",
            "",
        ]
    )
    if not embedded:
        lines.extend(
            [
                "## Limitations",
                "",
                *(f"- {item}" for item in artifact.get("limitations", [])),
                "",
            ]
        )
    return lines


def build_action_evaluation_markdown(artifact: Mapping[str, Any]) -> str:
    """Build the standalone Phase 6 report from the evaluated artifact."""
    return "\n".join(
        [
            "# Explanation and Action Evaluation",
            "",
            *action_evaluation_markdown_lines(artifact),
        ]
    )
