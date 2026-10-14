"""Phase 8: fixed-horizon maturity, provenance, and prospective isolation."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from sentinel.cross_project.splitting import build_leave_one_project_out_folds
from sentinel.features import build_file_snapshots, export_features
from sentinel.ml.data import (
    InvalidDatasetError,
    MODEL_FEATURE_COLUMNS,
    load_dataset,
    validate_dataset,
)
from sentinel.ml.splitting import TemporalSplits, chronological_split
from sentinel.ml.temporal import (
    audit_mature_partitions,
    partition_mature_data,
    temporal_markdown_lines,
)
from sentinel.ml.training import train_baselines
from sentinel.provenance import PROVENANCE_KEY, metadata_path, utc_day
from temporal_fixtures import attest, write_fixture
from test_ml import make_feature_data
from test_features import history_row

CUTS = {"validation_start": "2021-07-01", "test_start": "2022-07-01"}


@pytest.fixture
def data():
    return attest(validate_dataset(make_feature_data()))


def test_known_maturity_counts_and_accounting(data):
    frames, context = partition_mature_data(data, **CUTS)
    assert {k: len(v) for k, v in frames.items()} == {
        "train": 60,
        "validation": 36,
        "test": 24,
    }
    assert context["excluded_rows"] == 24
    assert sum(context["retained_rows"].values()) + context["excluded_rows"] == len(
        data
    )
    assert all(audit_mature_partitions(frames, context)["checks"].values())


@pytest.mark.parametrize(
    "boundary,partition", [("2021-07-01", "train"), ("2022-07-01", "validation")]
)
def test_day_90_is_purged_day_91_is_mature(data, boundary, partition):
    dates = [pd.Timestamp(boundary) - pd.Timedelta(days=n) for n in (90, 91)]
    extra = pd.concat([data.iloc[:4].assign(snapshot_date=date) for date in dates])
    expanded = attest(validate_dataset(pd.concat([data, extra], ignore_index=True)))
    frames, _ = partition_mature_data(expanded, **CUTS)
    assert dates[0] not in set(frames[partition].snapshot_date)
    assert dates[1] in set(frames[partition].snapshot_date)


def test_observation_day_90_is_inclusive_and_common_end(data):
    observed = attest(data, "2023-03-31")
    frames, context = partition_mature_data(observed, **CUTS, test_end="2022-12-31")
    assert frames["test"].snapshot_date.max() == pd.Timestamp("2022-12-31")
    assert context["test_end"] == "2022-12-31"
    with pytest.raises(InvalidDatasetError, match="coverage"):
        partition_mature_data(attest(data, "2023-03-30"), **CUTS, test_end="2022-12-31")


@pytest.mark.parametrize(
    "value,expected",
    [("2024-03-01T00:30:00+02:00", "2024-02-29"), ("2024-01-01", "2024-01-01")],
)
def test_utc_calendar_dates(value, expected):
    assert utc_day(value) == pd.Timestamp(expected)
    assert pd.Timestamp("2024-02-29") + pd.Timedelta(days=90) == pd.Timestamp(
        "2024-05-29"
    )


@pytest.mark.parametrize("value", [None, "bad", "NaT", float("inf")])
def test_invalid_boundaries(data, value):
    with pytest.raises(InvalidDatasetError):
        partition_mature_data(
            data, validation_start=value, test_start=CUTS["test_start"]
        )


def test_legacy_load_allowed_but_training_fails_closed(tmp_path):
    path = tmp_path / "legacy.csv"
    make_feature_data().to_csv(path, index=False)
    loaded = load_dataset(path)
    with pytest.raises(InvalidDatasetError, match="legacy/unverified"):
        chronological_split(loaded)
    with pytest.raises(InvalidDatasetError, match="legacy/unverified"):
        train_baselines(TemporalSplits(loaded, loaded, loaded))
    assert "legacy/unverified" in " ".join(temporal_markdown_lines(None))


@pytest.mark.parametrize(
    "key,value",
    [
        ("horizon_days", 89),
        ("target", "other"),
        ("schema_version", 2),
        ("date_convention", "local"),
        ("generator_version", None),
    ],
)
def test_incompatible_metadata_rejected(data, key, value):
    data.attrs[PROVENANCE_KEY][key] = value
    with pytest.raises(InvalidDatasetError, match="Incompatible"):
        chronological_split(data)


def test_sidecar_roundtrip_manifest_and_tamper(tmp_path):
    path = tmp_path / "features.csv"
    write_fixture(make_feature_data(), path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sidecars": {"owner/repository": metadata_path(path).name},
            }
        )
    )
    assert (
        chronological_split(
            load_dataset(path, dataset_metadata=manifest)
        ).temporal_context["protocol"]
        == "mature_labels_v1"
    )
    path.write_text(path.read_text().replace("999", "998"))
    with pytest.raises(InvalidDatasetError, match="CSV hash mismatch"):
        load_dataset(path)


@pytest.mark.parametrize("mutation", ["target", "feature", "endpoint", "repo", "hash"])
def test_attached_metadata_integrity(data, mutation):
    if mutation == "target":
        data.loc[0, "defect_next_90_days"] ^= 1
    elif mutation == "feature":
        data.loc[0, "commit_count"] += 1
    elif mutation == "endpoint":
        data.attrs[PROVENANCE_KEY]["repositories"]["owner/repository"][
            "observation_end"
        ] = None
    elif mutation == "repo":
        data.attrs[PROVENANCE_KEY]["repositories"] = {}
    else:
        data.attrs[PROVENANCE_KEY]["repositories"]["owner/repository"][
            "source_history_sha256"
        ] = "bad"
    with pytest.raises(InvalidDatasetError):
        chronological_split(data)


def test_live_partition_tamper_rejected_before_fit(data, monkeypatch):
    splits = chronological_split(data)
    splits.train.loc[0, "commit_count"] += 1
    monkeypatch.setattr(
        "sentinel.ml.training.build_model_pipelines",
        lambda *a, **kw: pytest.fail("fit must not start"),
    )
    with pytest.raises(InvalidDatasetError, match="rows_unchanged"):
        train_baselines(splits)


def test_no_adaptive_rescue_for_short_history_or_one_class(data):
    with pytest.raises(InvalidDatasetError, match="empty or missing"):
        chronological_split(attest(validate_dataset(make_feature_data(date_count=10))))
    data["defect_next_90_days"] = 0
    splits = chronological_split(attest(data))
    with pytest.raises(InvalidDatasetError, match="both target classes"):
        train_baselines(splits)


def test_partition_boundaries_independent_of_labels_and_features(data):
    original = chronological_split(data)
    changed = data.copy()
    changed["defect_next_90_days"] = 1 - changed["defect_next_90_days"]
    changed[MODEL_FEATURE_COLUMNS] *= 100
    altered = chronological_split(attest(changed))
    for (_, a), (_, b) in zip(original.frames(), altered.frames()):
        pd.testing.assert_frame_equal(
            a[["repository", "file_path", "snapshot_date"]],
            b[["repository", "file_path", "snapshot_date"]],
        )


def test_lopo_shared_calendar_and_no_future_sources(data):
    other = data.assign(repository="other/repository")
    combined = attest(validate_dataset(pd.concat([data, other], ignore_index=True)))
    folds = build_leave_one_project_out_folds(combined, **CUTS)
    assert len(folds) == 2
    for fold in folds:
        assert (
            fold.train.snapshot_date + pd.Timedelta(days=90)
            < pd.Timestamp(CUTS["validation_start"])
        ).all()
        assert (
            fold.validation.snapshot_date + pd.Timedelta(days=90)
            < fold.test.snapshot_date.min()
        ).all()
        assert set(fold.test.repository) == {fold.held_out_repository}
        assert fold.held_out_repository not in set(fold.train.repository) | set(
            fold.validation.repository
        )
        assert fold.temporal_context["validation_start"] == CUTS["validation_start"]
    with pytest.raises(InvalidDatasetError, match="explicit"):
        build_leave_one_project_out_folds(combined)
    with pytest.raises(InvalidDatasetError, match="legacy/unverified"):
        build_leave_one_project_out_folds(combined, train_fraction=0.8, **CUTS)


def test_lopo_missing_source_history_fails_without_moving_boundaries(data):
    late = data.loc[data.snapshot_date >= "2022-01-01"].assign(
        repository="late/repository"
    )
    combined = attest(validate_dataset(pd.concat([data, late], ignore_index=True)))
    with pytest.raises(InvalidDatasetError, match="missing required repositories"):
        build_leave_one_project_out_folds(combined, **CUTS)


def test_future_and_purged_rows_cannot_change_model_or_threshold(data):
    first_split = chronological_split(data, **CUTS)
    first = train_baselines(first_split, random_state=47)
    changed = data.copy()
    retained_dates = set(first_split.train.snapshot_date) | set(
        first_split.validation.snapshot_date
    )
    mask = ~changed.snapshot_date.isin(retained_dates)
    changed.loc[mask, MODEL_FEATURE_COLUMNS] *= 999
    changed.loc[mask, "defect_next_90_days"] = (
        1 - changed.loc[mask, "defect_next_90_days"]
    )
    second_split = chronological_split(attest(changed), **CUTS)
    pd.testing.assert_frame_equal(first_split.train, second_split.train)
    pd.testing.assert_frame_equal(first_split.validation, second_split.validation)
    second = train_baselines(second_split, random_state=47)
    assert first.best_model_name == second.best_model_name
    for name in first.models:
        assert (
            first.models[name].selected_threshold
            == pytest.approx(second.models[name].selected_threshold, rel=0, abs=1e-12)
        )
        np.testing.assert_allclose(
            first.models[name].pipeline.predict_proba(
                first_split.validation[MODEL_FEATURE_COLUMNS]
            ),
            second.models[name].pipeline.predict_proba(
                first_split.validation[MODEL_FEATURE_COLUMNS]
            ),
        )
        assert (
            first.models[name].feature_importance
            == second.models[name].feature_importance
        )


@pytest.mark.parametrize(
    "event_date,unchanged",
    [("2021-07-15", ("train",)), ("2022-07-15", ("train", "validation"))],
)
def test_history_metadata_and_future_bugfix_counterfactual(
    tmp_path, event_date, unchanged
):
    history = pd.DataFrame(
        [
            history_row(
                commit_hash="a",
                commit_date="2020-01-01",
                file_path="A.java",
                message="add",
            ),
            history_row(
                commit_hash="b",
                commit_date=event_date,
                file_path="A.java",
                message="cleanup",
            ),
            history_row(
                commit_hash="c",
                commit_date="2023-06-01",
                file_path="A.java",
                message="cleanup",
            ),
        ]
    )
    before = build_file_snapshots(history)
    history.loc[1, "commit_message"] = "fix defect"
    after = build_file_snapshots(history)
    a, _ = partition_mature_data(before, **CUTS)
    b, _ = partition_mature_data(after, **CUTS)
    # Changes after the training label deadline cannot alter retained training evidence.
    for partition in unchanged:
        pd.testing.assert_frame_equal(a[partition], b[partition])
    assert (before.defect_next_90_days != after.defect_next_90_days).any()
    path = tmp_path / "features.csv"
    export_features(after, path)
    loaded = load_dataset(path)
    assert (
        loaded.attrs[PROVENANCE_KEY]["repositories"]["owner/repository"][
            "observation_end"
        ]
        == "2023-06-01"
    )
    assert metadata_path(path).exists()


def test_audit_rejects_incompatible_context(data):
    frames, context = partition_mature_data(data, **CUTS)
    context = deepcopy(context)
    context["source_provenance"]["horizon_days"] = 30
    with pytest.raises(InvalidDatasetError, match="Incompatible"):
        audit_mature_partitions(frames, context)


@pytest.mark.parametrize(
    "field", ["source_history_sha256", "feature_frame_sha256", "observation_end"]
)
def test_missing_pre_fit_provenance_never_passes(data, field):
    frames, context = partition_mature_data(data, **CUTS)
    del context["source_provenance"]["repositories"]["owner/repository"][field]
    with pytest.raises(InvalidDatasetError):
        audit_mature_partitions(frames, context)


def test_common_observation_end_is_minimum_not_last_feature_date(data):
    combined = attest(
        validate_dataset(
            pd.concat([data, data.assign(repository="other")], ignore_index=True)
        )
    )
    combined.attrs[PROVENANCE_KEY]["repositories"]["other"]["observation_end"] = (
        "2023-02-28"
    )
    folds = build_leave_one_project_out_folds(combined, **CUTS)
    assert {fold.temporal_context["test_end"] for fold in folds} == {"2022-11-30"}
    for fold in folds:
        assert fold.test.snapshot_date.max() == pd.Timestamp("2022-11-30")
        reasons = {entry["reason"] for entry in fold.temporal_context["exclusions"]}
        assert "held_out_history" in reasons
        assert "source_at_or_after_test_start" in reasons


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(extra=float("nan")),
        lambda m: m.update(generator_version=float("inf")),
    ],
)
def test_nonstandard_json_metadata_rejected(data, mutate):
    mutate(data.attrs[PROVENANCE_KEY])
    with pytest.raises(InvalidDatasetError):
        chronological_split(data)


@pytest.mark.parametrize("command", ["train", "within-project", "cross-project"])
def test_cli_missing_metadata_has_no_success_or_outputs(tmp_path, capsys, command):
    from sentinel.evaluation.cli import main as evaluate_main
    from sentinel.ml.cli import main as train_main

    path = tmp_path / "legacy.csv"
    make_feature_data().to_csv(path, index=False)
    destination = tmp_path / "out"
    args = [str(path), "--output-dir", str(destination)]
    if command == "train":
        result = train_main(args)
    elif command == "within-project":
        result = evaluate_main([command, *args])
    else:
        other = tmp_path / "other.csv"
        make_feature_data().assign(repository="other").to_csv(other, index=False)
        result = evaluate_main(
            [
                command,
                str(other),
                *args,
                "--validation-start",
                CUTS["validation_start"],
                "--test-start",
                CUTS["test_start"],
            ]
        )
    captured = capsys.readouterr()
    assert result == 1
    assert "legacy/unverified" in captured.err
    assert "PASS" not in captured.out
    assert not destination.exists()


def test_v3_cli_verified_metadata(tmp_path, capsys):
    from sentinel.ml.cli import main

    path = tmp_path / "features.csv"
    write_fixture(make_feature_data(), path)
    destination = tmp_path / "trained"
    assert (
        main(
            [
                str(path),
                "--output-dir",
                str(destination),
                "--validation-start",
                CUTS["validation_start"],
                "--test-start",
                CUTS["test_start"],
            ]
        )
        == 0
    )
    assert "90-day label maturity" in capsys.readouterr().out
    artifact = json.loads(
        (destination / "metadata.json").read_text(),
        parse_constant=lambda value: pytest.fail(value),
    )
    audit = artifact["leakage_audit"]["temporal_maturity"]
    assert audit["status"] == "passed"
    assert audit["retained_rows"] == {"train": 60, "validation": 36, "test": 24}


@pytest.mark.parametrize(
    "change,available",
    [("none", True), ("legacy", False), ("schedule", False), ("population", False)],
)
def test_same_project_comparison_requires_matched_protocol_and_population(
    data, tmp_path, change, available
):
    from types import SimpleNamespace
    from sentinel.cross_project.experiment import _same_project_comparison

    common = attest(data.assign(repository="apache/commons-lang"))
    combined = attest(validate_dataset(pd.concat([common, data], ignore_index=True)))
    fold = next(
        f
        for f in build_leave_one_project_out_folds(combined, **CUTS)
        if f.held_out_repository == "apache/commons-lang"
    )
    splits = chronological_split(common, **CUTS)
    audit = audit_mature_partitions(dict(splits.frames()), splits.temporal_context)
    if change == "legacy":
        audit = {}
    elif change == "schedule":
        audit["test_start"] = "2022-08-01"
    elif change == "population":
        audit["partition_sha256"]["test"] = "0" * 64
    metadata = {
        "leakage_audit": {"temporal_maturity": audit},
        "all_model_comparison": {
            "logistic_regression": {"test": {"pr_auc": 0.5, "f1": 0.5, "roc_auc": 0.5}}
        },
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    rows = pd.DataFrame(
        [
            {
                "held_out_repository": "apache/commons-lang",
                "model": "logistic_regression",
                "threshold_strategy": "validation_selected",
                "pr_auc": 0.4,
                "f1": 0.4,
                "roc_auc": 0.4,
            }
        ]
    )
    result = _same_project_comparison(path, rows, [SimpleNamespace(fold=fold)])
    assert result["available"] is available
    if not available:
        assert "comparisons" not in result


def test_split_determinism_under_input_reordering(data):
    first = chronological_split(data)
    second = chronological_split(data.sample(frac=1, random_state=19))
    assert first.temporal_context == second.temporal_context
    for (_, a), (_, b) in zip(first.frames(), second.frames()):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize(
    "option,value",
    [
        ("validation_start", "2022-07-01"),
        ("test_end", "2022-06-30"),
        ("test_start", "bad"),
    ],
)
def test_invalid_calendar_order_fails(data, option, value):
    cuts = {**CUTS, option: value}
    with pytest.raises(InvalidDatasetError):
        partition_mature_data(data, **cuts)


def test_partial_within_dates_and_missing_lopo_cli_cuts(tmp_path, capsys, data):
    from sentinel.cross_project.cli import main

    with pytest.raises(InvalidDatasetError, match="both"):
        chronological_split(data, validation_start="2021-07-01")
    paths = [tmp_path / "a.csv", tmp_path / "b.csv"]
    for index, path in enumerate(paths):
        write_fixture(data.assign(repository=f"repo-{index}"), path)
    assert (
        main([*(str(path) for path in paths), "--output-dir", str(tmp_path / "out")])
        == 1
    )
    assert "explicit validation_start and test_start" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["intraday", "fractional_label"])
def test_pre_fit_rejects_noncanonical_partition_values(data, mutation):
    splits = chronological_split(data)
    if mutation == "intraday":
        splits.train.loc[0, "snapshot_date"] += pd.Timedelta(hours=1)
    else:
        splits.train["defect_next_90_days"] = splits.train.defect_next_90_days.astype(
            float
        )
        splits.train.loc[0, "defect_next_90_days"] = 0.5
    with pytest.raises(InvalidDatasetError):
        train_baselines(splits)
