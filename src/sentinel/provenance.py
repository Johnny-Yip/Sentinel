"""Hash-bound observation coverage for the fixed V2 90-day target.

Metadata attests to the supplied mining history, not to externally verified defect
ground truth. Missing metadata is never inferred from the last feature snapshot.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from sentinel import __version__

LABEL_WINDOW_DAYS = 90
TARGET = "defect_next_90_days"
DATE_CONVENTION = "UTC_calendar_day; features_through_day; label_(T,T+90d]"
PROVENANCE_KEY = "sentinel_feature_metadata"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_day(value: Any) -> pd.Timestamp:
    """Parse a finite date/timestamp, normalize to its UTC day."""
    try:
        if isinstance(value, (int, float, bool)):
            raise ValueError("numeric dates are ambiguous")
        day = pd.to_datetime(value, utc=True, errors="raise")
        if not isinstance(day, pd.Timestamp) or pd.isna(day):
            raise ValueError("not a finite timestamp")
        return day.normalize().tz_localize(None)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid UTC date: {value!r}") from exc


def feature_digest(frame: pd.DataFrame) -> str:
    """Canonical identity of the model inputs, metadata keys and target rows."""
    from sentinel.ml.data import MODEL_FEATURE_COLUMNS, REQUIRED_COLUMNS

    canonical = frame.loc[:, REQUIRED_COLUMNS].copy()
    canonical["snapshot_date"] = pd.to_datetime(
        canonical["snapshot_date"], utc=True
    ).dt.strftime("%Y-%m-%d")
    for column in MODEL_FEATURE_COLUMNS:
        canonical[column] = canonical[column].astype(float)
    canonical[TARGET] = canonical[TARGET].astype(float)
    canonical = canonical.sort_values(["repository", "snapshot_date", "file_path"])
    return hashlib.sha256(
        canonical.to_csv(
            index=False, float_format="%.17g", lineterminator="\n"
        ).encode()
    ).hexdigest()


def build_feature_metadata(
    history: pd.DataFrame,
    features: pd.DataFrame,
    *,
    source_history_sha256: str | None = None,
) -> dict[str, Any]:
    """Record coverage from mining history; do not derive it from feature dates."""
    days = pd.to_datetime(history["commit_date"], utc=True).dt.normalize()
    endpoints = days.groupby(history["repository"]).max()
    history_hash = (
        source_history_sha256
        or hashlib.sha256(
            history.to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest()
    )
    return {
        "schema_version": 1,
        "generator_version": __version__,
        "target": TARGET,
        "horizon_days": LABEL_WINDOW_DAYS,
        "date_convention": DATE_CONVENTION,
        "history_hash_basis": "source_file"
        if source_history_sha256
        else "dataframe_csv",
        "repositories": {
            str(repository): {
                "observation_end": endpoints[repository].strftime("%Y-%m-%d"),
                "source_history_sha256": history_hash,
                "feature_frame_sha256": feature_digest(rows),
                "row_count": len(rows),
            }
            for repository, rows in features.groupby("repository", sort=True)
        },
    }


def validate_metadata_contract(metadata: Any, expected_repositories) -> None:
    """Shared fail-closed structural check, also used by pre-fit audits."""
    from sentinel.ml.data import InvalidDatasetError

    if not isinstance(metadata, dict):
        raise InvalidDatasetError(
            "Missing feature provenance: legacy/unverified input cannot pass a "
            "prospective audit. Generate a feature metadata sidecar from source history."
        )
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or metadata.get("target") != TARGET
        or metadata.get("horizon_days") != LABEL_WINDOW_DAYS
        or metadata.get("date_convention") != DATE_CONVENTION
        or not isinstance(metadata.get("generator_version"), str)
        or not metadata.get("generator_version")
        or metadata.get("history_hash_basis") not in ("source_file", "dataframe_csv")
    ):
        raise InvalidDatasetError(
            "Incompatible feature provenance target/horizon/schema/date convention."
        )
    try:
        json.dumps(metadata, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise InvalidDatasetError(
            "Feature provenance must contain finite JSON values."
        ) from exc
    repositories = metadata.get("repositories")
    if not isinstance(repositories, dict) or set(repositories) != set(
        expected_repositories
    ):
        raise InvalidDatasetError(
            "Feature provenance repository coverage does not match input."
        )
    for entry in repositories.values():
        if (
            not isinstance(entry, dict)
            or type(entry.get("row_count")) is not int
            or entry["row_count"] <= 0
        ):
            raise InvalidDatasetError("Malformed repository provenance.")
        for key in ("source_history_sha256", "feature_frame_sha256"):
            if not isinstance(entry.get(key), str) or not re.fullmatch(
                r"[0-9a-f]{64}", entry[key]
            ):
                raise InvalidDatasetError(f"Invalid provenance hash: {key}.")
        try:
            utc_day(entry.get("observation_end"))
        except ValueError as exc:
            raise InvalidDatasetError(
                "Missing or invalid observation endpoint."
            ) from exc


def verify_feature_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate attached coverage and row hashes; return a defensive copy."""
    from sentinel.ml.data import InvalidDatasetError

    metadata = frame.attrs.get(PROVENANCE_KEY)
    validate_metadata_contract(metadata, frame["repository"].unique())
    for repository, rows in frame.groupby("repository", sort=True):
        entry = metadata["repositories"][repository]
        if entry.get("row_count") != len(rows) or entry[
            "feature_frame_sha256"
        ] != feature_digest(rows):
            raise InvalidDatasetError(
                f"Feature provenance hash/row count mismatch for {repository}."
            )
    return deepcopy(metadata)


def metadata_path(feature_csv: str | Path) -> Path:
    return Path(str(feature_csv) + ".metadata.json")


def save_feature_metadata(frame: pd.DataFrame, feature_csv: str | Path) -> Path:
    metadata = verify_feature_metadata(frame)
    metadata["feature_csv_sha256"] = sha256_file(feature_csv)
    destination = metadata_path(feature_csv)
    destination.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_feature_metadata(
    frame: pd.DataFrame,
    feature_csv: Path,
    metadata_source: str | Path | None,
) -> pd.DataFrame:
    """Read a sidecar or a manifest mapping repository names to sidecar paths."""
    from sentinel.ml.data import InvalidDatasetError

    source = (
        Path(metadata_source).expanduser().resolve()
        if metadata_source
        else metadata_path(feature_csv)
    )
    if not source.exists() and metadata_source is None:
        # Loading/inspecting legacy CSVs remains possible; splitting will reject them.
        return frame
    try:
        metadata = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("expected an object")
        if "sidecars" in metadata:
            if metadata.get("schema_version") != 1 or not isinstance(
                metadata["sidecars"], dict
            ):
                raise ValueError("invalid manifest")
            paths = {
                metadata["sidecars"][name] for name in frame["repository"].unique()
            }
            if len(paths) != 1:
                raise ValueError(
                    "all repositories in one CSV must reference its same sidecar"
                )
            source = (source.parent / next(iter(paths))).resolve()
            metadata = json.loads(source.read_text(encoding="utf-8"))
        if metadata.get("feature_csv_sha256") != sha256_file(feature_csv):
            raise ValueError("feature CSV hash mismatch")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InvalidDatasetError(
            f"Invalid feature provenance {source}: {exc}"
        ) from exc
    frame.attrs[PROVENANCE_KEY] = metadata
    verify_feature_metadata(frame)
    return frame
