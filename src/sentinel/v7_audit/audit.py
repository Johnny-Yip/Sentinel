"""Deterministic dataset and candidate-schedule diagnostics, separate from V6.

This module describes eligibility only. It does not implement a training split
interface, fit estimators, choose boundaries, or expose target-test class counts
in fold diagnostics. Dataset-wide label summaries are descriptive inventory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel.ml.data import InvalidDatasetError, load_dataset, validate_dataset
from sentinel.provenance import (
    LABEL_WINDOW_DAYS, TARGET, sha256_file, utc_day,
    verify_feature_metadata,
)

KEYS = ["repository", "file_path", "snapshot_date"]
ROLES = ("model_fitting", "model_selection", "calibration_fitting", "policy_selection")
HORIZON = pd.Timedelta(days=LABEL_WINDOW_DAYS)
LIMITATIONS = [
    "Commit-message bug-fix involvement is a proxy, not verified defect ground truth.",
    "Deleted/renamed paths are carried forward; the V1 schema lacks lifecycle events.",
    "Repeated files and overlapping 90-day windows are dependent observations.",
    "Positive snapshots and positive dates are not independent bug-fix event counts.",
    "Provenance binds supplied inputs, not upstream extraction completeness.",
]


@dataclass(frozen=True)
class CandidateSchedule:
    model_selection_start: str = "2022-06-30"
    calibration_start: str = "2023-09-30"
    policy_start: str = "2024-09-30"
    test_start: str = "2025-12-31"
    test_end: str = "2026-03-31"

    def days(self) -> tuple[pd.Timestamp, ...]:
        days = tuple(utc_day(v) for v in asdict(self).values())
        if not (days[0] < days[1] < days[2] < days[3] <= days[4]):
            raise InvalidDatasetError("Candidate boundaries must be strictly ordered before test_end.")
        return days


def counts(frame: pd.DataFrame, *, labels: bool = True) -> dict[str, Any]:
    """Native JSON scalars; no undefined rates manufactured as zero."""
    n = len(frame)
    result = {
        "rows": n,
        "snapshot_dates": int(frame.snapshot_date.nunique()),
        "unique_files": len(frame[["repository", "file_path"]].drop_duplicates()),
        "start": frame.snapshot_date.min().strftime("%Y-%m-%d") if n else None,
        "end": frame.snapshot_date.max().strftime("%Y-%m-%d") if n else None,
    }
    if labels:
        positives = int(frame[TARGET].sum())
        result.update(positives=positives, negatives=n - positives,
                      prevalence=positives / n if n else None,
                      positive_dates=int(frame.loc[frame[TARGET].eq(1), "snapshot_date"].nunique()))
    return result


def repository_summary(frame: pd.DataFrame, observation_end: str) -> dict[str, Any]:
    dates = frame.snapshot_date.drop_duplicates().sort_values()
    gaps = dates.diff().dt.days
    month_ends = dates.loc[dates.dt.is_month_end]
    quarters = dates.loc[dates.dt.is_quarter_end]
    periods = []
    for month, rows in frame.groupby(frame.snapshot_date.dt.to_period("M"), sort=True):
        periods.append({"month": str(month), **counts(rows)})
    quarter_counts = []
    quarter_rows = frame.loc[frame.snapshot_date.dt.is_quarter_end]
    for quarter, rows in quarter_rows.groupby(quarter_rows.snapshot_date.dt.to_period("Q"), sort=True):
        quarter_counts.append({"quarter": str(quarter), **counts(rows)})
    path_dates = frame.sort_values(["file_path", "snapshot_date"]).groupby("file_path").snapshot_date.diff().dt.days
    return {
        **counts(frame), "observation_end": observation_end,
        "label_windows_within_observation": int((frame.snapshot_date + HORIZON <= utc_day(observation_end)).sum()),
        "monthly_coverage": {
            "observed_month_end_dates": len(month_ends),
            "possible_month_end_dates": len(pd.date_range(dates.min(), dates.max(), freq="ME")),
            "by_month": periods,
        },
        "quarterly_coverage": {
            "observed_quarter_end_dates": len(quarters),
            "possible_quarter_end_dates": len(pd.date_range(dates.min(), dates.max(), freq="QE")),
            "quarter_end_snapshots_only": quarter_counts,
        },
        "dependence": {
            "files_observed_more_than_once": int(frame.groupby("file_path").size().gt(1).sum()),
            "rows_beyond_first_file_observation": len(frame) - int(frame.file_path.nunique()),
            "adjacent_file_windows_overlapping": int(path_dates.lt(LABEL_WINDOW_DAYS).sum()),
            "adjacent_date_gaps_under_90_days": int(gaps.lt(LABEL_WINDOW_DAYS).sum()),
            "interpretation": "descriptive dependence indicators, not effective sample size",
        },
    }


def audit_candidate(
    data: pd.DataFrame, schedule: CandidateSchedule = CandidateSchedule(), *,
    minimum_class_count: int = 1,
) -> dict[str, Any]:
    """Report whole-group eligibility; count labels on source development only.

    A two-class support gate is a necessary feasibility check, never a claim of
    sufficient statistical power. Thresholds are declared inputs, not adapted.
    """
    if type(minimum_class_count) is not int or minimum_class_count < 1:
        raise ValueError("minimum_class_count must be a positive integer")
    data = validate_dataset(data)
    metadata = verify_feature_metadata(data)
    s, c, p, t, e = schedule.days()
    repositories = sorted(metadata["repositories"])
    if len(repositories) < 2:
        raise InvalidDatasetError("Candidate LOPO requires at least two repositories.")
    observation = {r: utc_day(v["observation_end"]) for r, v in metadata["repositories"].items()}
    dates = data.snapshot_date
    mature = dates + HORIZON
    observed = mature <= data.repository.map(observation)
    folds = []
    for target in repositories:
        source = data.repository.ne(target)
        role = pd.Series("excluded", index=data.index, dtype=object)
        reason = pd.Series("", index=data.index, dtype=object)
        reason.loc[~source & dates.lt(t)] = "held_out_history"
        reason.loc[source & dates.ge(t)] = "source_at_or_after_test_start"
        stage_reports = {}
        for name, lower, upper in zip(ROLES, (None, s, c, p), (s, c, p, t), strict=True):
            candidate = source & dates.lt(upper)
            if lower is not None:
                candidate &= dates.ge(lower)
            eligible = candidate & mature.lt(upper) & observed
            role.loc[eligible] = name
            reason.loc[candidate & ~mature.lt(upper)] = "immature_" + name + "_label"
            reason.loc[candidate & ~observed] = "incomplete_development_label"
            summaries = {}
            for repository in repositories:
                if repository == target:
                    continue
                summaries[repository] = counts(data.loc[eligible & data.repository.eq(repository)])
            aggregate = counts(data.loc[eligible])
            problems = []
            if not aggregate["rows"]:
                problems.append("empty_development_partition")
            for repository, summary in summaries.items():
                if not summary["rows"]:
                    problems.append("missing_source_repository:" + repository)
                elif min(summary["positives"], summary["negatives"]) < minimum_class_count:
                    problems.append("insufficient_repository_class_support:" + repository)
            if min(aggregate["positives"], aggregate["negatives"]) < minimum_class_count:
                problems.append("insufficient_pooled_class_support")
            stage_reports[name] = {
                "pooled": aggregate, "repositories": summaries, "issues": problems,
                "pooled_two_class_gate_passed": aggregate["rows"] > 0
                and min(aggregate["positives"], aggregate["negatives"]) >= minimum_class_count,
                "identity_sha256": identity_digest(data.loc[eligible]),
            }
        test_candidate = ~source & dates.ge(t)
        test_eligible = test_candidate & dates.le(e) & observed
        role.loc[test_eligible] = "final_evaluation"
        reason.loc[test_candidate & dates.gt(e)] = "after_test_end"
        reason.loc[test_candidate & ~observed] = "incomplete_test_label"
        missing_reason = role.eq("excluded") & reason.eq("")
        if missing_reason.any():
            raise AssertionError("Unaccounted candidate rows")
        excluded = data.loc[role.eq("excluded"), ["repository", "snapshot_date"]].copy()
        excluded["reason"] = reason.loc[excluded.index]
        exclusions = [
            {"repository": r, "reason": why, "rows": len(rows),
             "start": rows.snapshot_date.min().strftime("%Y-%m-%d"),
             "end": rows.snapshot_date.max().strftime("%Y-%m-%d")}
            for (r, why), rows in excluded.groupby(["repository", "reason"], sort=True)
        ]
        assignment = data[KEYS].copy()
        assignment["role"] = role
        checks = {
            "held_out_excluded_from_development": not bool((~source & role.isin(ROLES)).any()),
            "whole_repository_date_groups": bool(assignment.groupby(["repository", "snapshot_date"]).role.nunique().eq(1).all()),
            "complete_row_accounting": int(role.ne("excluded").sum()) + len(excluded) == len(data),
            "exclusion_accounting": sum(v["rows"] for v in exclusions) == len(excluded),
            "eligible_labels_observed": bool(observed.loc[role.ne("excluded")].all()),
            "requested_common_test_coverage": e + HORIZON <= min(observation.values()),
        }
        folds.append({
            "held_out_repository": target, "source_repositories": [r for r in repositories if r != target],
            "development": stage_reports,
            "final_evaluation": {**counts(data.loc[test_eligible], labels=False),
                                 "labels": "withheld_from_candidate_audit",
                                 "identity_sha256": identity_digest(data.loc[test_eligible])},
            "input_rows": len(data), "excluded_rows": len(excluded), "exclusions": exclusions,
            "checks": checks,
            "structural_status": "passed" if all(checks.values()) and test_eligible.any()
            and all(v["pooled"]["rows"] and all(r["rows"] for r in v["repositories"].values()) for v in stage_reports.values()) else "failed",
            "pooled_class_support_passed": all(v["pooled_two_class_gate_passed"] for v in stage_reports.values()),
        })
    return {
        "status": "candidate_only", "schedule": asdict(schedule), "horizon_days": LABEL_WINDOW_DAYS,
        "maturity_rule": "label_end < next_role_start; final label_end <= observation_end",
        "minimum_class_count": minimum_class_count,
        "support_gate_limitation": "Two-class support is necessary, not proof of adequate effective sample size.",
        "test_labels_used_for_selection": False, "folds": folds,
    }


def identity_digest(frame: pd.DataFrame) -> str:
    """Label-free fingerprint: target outcomes cannot affect schedule evidence."""
    import hashlib
    keys = frame[KEYS].sort_values(KEYS).copy()
    keys["snapshot_date"] = keys.snapshot_date.dt.strftime("%Y-%m-%d")
    return hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def run_audit(paths: list[Path], root: Path, schedule: CandidateSchedule = CandidateSchedule()) -> dict[str, Any]:
    """Inspect files; invalid inputs produce explicit failures and no fold audit."""
    reports, frames, seen = [], [], set()
    for path in sorted(paths):
        report: dict[str, Any] = {"path": relative_path(path, root), "status": "invalid"}
        try:
            raw = pd.read_csv(path, usecols=KEYS)
            # Normalize dates like V6 before counting collisions.
            raw["snapshot_date"] = pd.to_datetime(raw.snapshot_date, utc=True, errors="raise").dt.normalize()
            report["duplicate_identity_rows"] = int(raw.duplicated(KEYS, keep=False).sum())
            report["duplicate_identity_excess"] = int(raw.duplicated(KEYS).sum())
            frame = load_dataset(path)
            metadata = verify_feature_metadata(frame)
            overlap = seen & set(metadata["repositories"])
            if overlap:
                raise InvalidDatasetError("Repositories repeated across inputs: " + ", ".join(sorted(overlap)))
            seen.update(metadata["repositories"])
            repository_reports = {}
            for repository, rows in frame.groupby("repository", sort=True):
                entry = metadata["repositories"][repository]
                history = root / (repository.rsplit("/", 1)[-1] + "_java_history.csv")
                history_status = "missing_locally"
                if history.exists():
                    history_status = "matched" if sha256_file(history) == entry["source_history_sha256"] else "hash_mismatch"
                repository_reports[repository] = {
                    **repository_summary(rows, entry["observation_end"]),
                    "source_history": {"path": relative_path(history, root), "status": history_status},
                }
                if history_status == "hash_mismatch":
                    raise InvalidDatasetError("Source history hash mismatch: " + repository)
            report.update(status="verified", csv_sha256=sha256_file(path), provenance=metadata,
                          repositories=repository_reports)
            frames.append(frame)
        except (InvalidDatasetError, OSError, ValueError, KeyError) as exc:
            report["error"] = str(exc)
        reports.append(report)
    valid = bool(reports) and all(r["status"] == "verified" for r in reports)
    candidate = None
    if valid:
        from copy import deepcopy
        from sentinel.provenance import PROVENANCE_KEY
        data = pd.concat(frames, ignore_index=True)
        metadata = deepcopy(frames[0].attrs[PROVENANCE_KEY])
        metadata.pop("feature_csv_sha256", None)
        metadata["repositories"] = {k: v for f in frames for k, v in f.attrs[PROVENANCE_KEY]["repositories"].items()}
        data.attrs[PROVENANCE_KEY] = metadata
        candidate = audit_candidate(data, schedule)
    return {"schema_version": 1, "scope": "V7 Phase 1 audit only", "input_status": "verified" if valid else "invalid",
            "repository_count": len(seen), "datasets": reports, "candidate": candidate,
            "known_limitations": LIMITATIONS,
            "label_access": "Inventory prevalence is descriptive; fold development uses source labels only; no target-test class statistics in candidate output."}


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())
