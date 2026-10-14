"""One mature-label protocol shared by within-project and prospective LOPO."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sentinel.ml.data import InvalidDatasetError, validate_dataset
from sentinel.provenance import (
    DATE_CONVENTION,
    TARGET,
    LABEL_WINDOW_DAYS,
    feature_digest,
    utc_day,
    verify_feature_metadata,
    validate_metadata_contract,
)

TEMPORAL_PROTOCOL = "mature_labels_v1"
HORIZON = pd.Timedelta(days=LABEL_WINDOW_DAYS)


def _day(value: Any) -> pd.Timestamp:
    try:
        return utc_day(value)
    except ValueError as exc:
        raise InvalidDatasetError(str(exc)) from exc


def partition_mature_data(
    data: pd.DataFrame,
    *,
    validation_start: Any,
    test_start: Any,
    test_end: Any = None,
    held_out_repository: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Freeze calendar boundaries, purge full date groups, and attest partitions.

    The model is locked before validation-start UTC day; selection is locked
    before test-start UTC day. No label or class count chooses a boundary.
    """
    data = validate_dataset(data)
    provenance = verify_feature_metadata(data)
    observations = {
        name: _day(entry["observation_end"])
        for name, entry in provenance["repositories"].items()
    }
    validation_start, test_start = _day(validation_start), _day(test_start)
    latest_test = min(observations.values()) - HORIZON
    test_end = latest_test if test_end is None else _day(test_end)
    if not validation_start < test_start <= test_end:
        raise InvalidDatasetError(
            "Require validation_start < test_start <= test_end with complete observation coverage."
        )
    if test_end > latest_test:
        raise InvalidDatasetError(
            "test_end exceeds common fully observed 90-day label coverage."
        )
    repositories = set(observations)
    if held_out_repository is not None and (
        held_out_repository not in repositories or len(repositories) < 2
    ):
        raise InvalidDatasetError(
            "LOPO requires a known held-out repository and at least one source repository."
        )
    dates = data["snapshot_date"]
    maturity = dates + HORIZON
    source = (
        pd.Series(True, index=data.index)
        if held_out_repository is None
        else data["repository"].ne(held_out_repository)
    )
    target = (
        pd.Series(True, index=data.index) if held_out_repository is None else ~source
    )
    candidates = {
        "train": source & (dates < validation_start),
        "validation": source & (dates >= validation_start) & (dates < test_start),
        "test": target & (dates >= test_start),
    }
    masks = {
        "train": candidates["train"] & (maturity < validation_start),
        "validation": candidates["validation"] & (maturity < test_start),
        "test": candidates["test"]
        & (dates <= test_end)
        & (maturity <= data["repository"].map(observations)),
    }
    reasons = pd.Series("", index=data.index, dtype=object)
    reasons.loc[candidates["train"] & ~masks["train"]] = "immature_training_label"
    reasons.loc[candidates["validation"] & ~masks["validation"]] = (
        "immature_validation_label"
    )
    reasons.loc[candidates["test"] & ~masks["test"]] = "after_test_end"
    reasons.loc[
        candidates["test"] & (maturity > data["repository"].map(observations))
    ] = "incomplete_test_label"
    if held_out_repository is not None:
        reasons.loc[target & (dates < test_start)] = "held_out_history"
        reasons.loc[source & (dates >= test_start)] = "source_at_or_after_test_start"
    frames = {
        name: data.loc[mask].copy().reset_index(drop=True)
        for name, mask in masks.items()
    }
    expected_sources = (
        repositories - {held_out_repository} if held_out_repository else repositories
    )
    for name, frame in frames.items():
        expected = (
            ({held_out_repository} if held_out_repository else repositories)
            if name == "test"
            else expected_sources
        )
        if frame.empty or set(frame["repository"].unique()) != expected:
            raise InvalidDatasetError(
                f"Maturity purge/coverage leaves {name} empty or missing required repositories. "
                "Use more history or explicit calendar boundaries; the 90-day rule is not relaxed."
            )
    excluded = data.loc[reasons.ne(""), ["repository", "snapshot_date"]].copy()
    excluded["reason"] = reasons[reasons.ne("")].to_numpy()
    exclusions = [
        {
            "repository": repository,
            "reason": reason,
            "rows": len(rows),
            "start_date": rows.snapshot_date.min().strftime("%Y-%m-%d"),
            "end_date": rows.snapshot_date.max().strftime("%Y-%m-%d"),
        }
        for (repository, reason), rows in excluded.groupby(
            ["repository", "reason"], sort=True
        )
    ]
    context = {
        "protocol": TEMPORAL_PROTOCOL,
        "horizon_days": LABEL_WINDOW_DAYS,
        "boundary_convention": "label_end < next_partition_start; UTC day; complete final day",
        "validation_start": validation_start.strftime("%Y-%m-%d"),
        "test_start": test_start.strftime("%Y-%m-%d"),
        "test_end": test_end.strftime("%Y-%m-%d"),
        "held_out_repository": held_out_repository,
        "source_provenance": provenance,
        "input_rows": len(data),
        "candidate_rows": {name: int(mask.sum()) for name, mask in candidates.items()},
        "retained_rows": {name: len(frame) for name, frame in frames.items()},
        "excluded_rows": len(excluded),
        "exclusions": exclusions,
        "partition_sha256": {
            name: feature_digest(frame) for name, frame in frames.items()
        },
    }
    audit_mature_partitions(frames, context)
    return frames, context


def audit_mature_partitions(
    frames: dict[str, pd.DataFrame],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recheck live partitions before fitting; unattested/manual splits fail closed."""
    if not isinstance(context, dict) or context.get("protocol") != TEMPORAL_PROTOCOL:
        raise InvalidDatasetError(
            "Missing temporal provenance: legacy/unverified splits cannot be used for prospective training."
        )
    try:
        validation_start = _day(context["validation_start"])
        test_start, test_end = _day(context["test_start"]), _day(context["test_end"])
        provenance = context["source_provenance"]
        observations = {
            name: _day(entry["observation_end"])
            for name, entry in provenance["repositories"].items()
        }
        validate_metadata_contract(provenance, observations)
        held_out = context["held_out_repository"]
        sources = set(observations) - {held_out} if held_out else set(observations)
        checks = {
            "compatible_provenance": provenance.get("schema_version") == 1
            and provenance.get("target") == TARGET
            and provenance.get("horizon_days") == LABEL_WINDOW_DAYS
            and provenance.get("date_convention") == DATE_CONVENTION,
            "fixed_90_day_horizon": context["horizon_days"] == LABEL_WINDOW_DAYS,
            "ordered_calendar_boundaries": validation_start < test_start <= test_end,
            "common_observation_coverage": test_end + HORIZON
            <= min(observations.values()),
            "source_row_accounting": sum(
                entry["row_count"] for entry in provenance["repositories"].values()
            )
            == context["input_rows"],
            "row_accounting": sum(len(frame) for frame in frames.values())
            + context["excluded_rows"]
            == context["input_rows"],
            "exclusion_accounting": sum(row["rows"] for row in context["exclusions"])
            == context["excluded_rows"],
        }
        keys, date_groups = [], []
        for name in ("train", "validation", "test"):
            frame = validate_dataset_partition(frames[name])
            dates = frame["snapshot_date"]
            label_end = dates + HORIZON
            expected = (
                ({held_out} if held_out else set(observations))
                if name == "test"
                else sources
            )
            checks[f"{name}_repository_coverage"] = (
                set(frame.repository.unique()) == expected
            )
            checks[f"{name}_rows_unchanged"] = (
                len(frame) == context["retained_rows"][name]
                and feature_digest(frame) == context["partition_sha256"][name]
            )
            checks[f"{name}_labels_observed"] = bool(
                (label_end <= frame.repository.map(observations)).all()
            )
            if name == "train":
                eligible = label_end < validation_start
            elif name == "validation":
                eligible = (dates >= validation_start) & (label_end < test_start)
            else:
                eligible = (dates >= test_start) & (dates <= test_end)
            checks[f"{name}_temporal_eligibility"] = bool(eligible.all())
            keys.append(
                set(
                    map(
                        tuple,
                        frame[["repository", "file_path", "snapshot_date"]].to_numpy(),
                    )
                )
            )
            date_groups.append(
                set(map(tuple, frame[["repository", "snapshot_date"]].to_numpy()))
            )
        checks["disjoint_sample_identities"] = all(
            not keys[i] & keys[j] for i in range(3) for j in range(i)
        )
        checks["whole_repository_date_groups"] = all(
            not date_groups[i] & date_groups[j] for i in range(3) for j in range(i)
        )
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise InvalidDatasetError(
            f"Invalid temporal audit context/partitions: {exc}"
        ) from exc
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise InvalidDatasetError(
            "Temporal maturity audit failed: " + ", ".join(failed)
        )
    return {**context, "status": "passed", "checks": checks}


def validate_dataset_partition(frame: pd.DataFrame) -> pd.DataFrame:
    """Partition checks without the loader's three-distinct-dates requirement."""
    if frame.empty:
        raise InvalidDatasetError("Empty temporal partition.")
    # Structural validation happened on the source dataset; fingerprints above
    # bind every retained value. Reject date coercions and duplicate identities.
    if not pd.api.types.is_datetime64_any_dtype(frame["snapshot_date"]):
        raise InvalidDatasetError("Partition dates must be normalized timestamps.")
    dates = frame["snapshot_date"]
    if (
        dates.dt.tz is not None
        or dates.isna().any()
        or not dates.eq(dates.dt.normalize()).all()
    ):
        raise InvalidDatasetError(
            "Partition dates must be normalized UTC days without a timezone."
        )
    if not frame["defect_next_90_days"].isin([0, 1]).all():
        raise InvalidDatasetError("Partition labels must be binary 0/1.")
    if frame.duplicated(["repository", "file_path", "snapshot_date"]).any():
        raise InvalidDatasetError("Duplicate temporal sample identity.")
    return frame


def temporal_markdown_lines(audit: dict[str, Any] | None) -> list[str]:
    """Shared report wording; old report objects cannot claim a prospective pass."""
    if (
        not audit
        or audit.get("protocol") != TEMPORAL_PROTOCOL
        or audit.get("status") != "passed"
    ):
        return [
            "Temporal maturity: **legacy/unverified**. This report does not establish prospective validity.",
            "",
        ]
    lines = [
        f"Temporal maturity: **passed** (`{TEMPORAL_PROTOCOL}`, 90-day outcome window).",
        f"- Validation start: {audit['validation_start']}; test start: {audit['test_start']}; test end: {audit['test_end']} (UTC days).",
        "- Training labels end strictly before validation start; validation labels end strictly before test start. Test labels have complete observed windows.",
        f"- Candidate rows: {audit['candidate_rows']}; retained rows: {audit['retained_rows']}; excluded rows: {audit['excluded_rows']}.",
        "- Observation endpoints: "
        + ", ".join(
            f"{name}: {entry['observation_end']}"
            for name, entry in audit["source_provenance"]["repositories"].items()
        ),
    ]
    lines.extend(
        f"- Excluded {row['rows']} rows from {row['repository']}: {row['reason']} ({row['start_date']} through {row['end_date']})."
        for row in audit["exclusions"]
    )
    lines.extend(
        [
            "- This audit verifies the declared retained-data protocol; it does not establish defect-label ground truth or independence of repeated snapshots.",
            "",
        ]
    )
    return lines
