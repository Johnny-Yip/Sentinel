"""V7 audits inspect evidence without fitting models or consulting test labels."""

from copy import deepcopy
import json

import pandas as pd
import pytest

from sentinel.ml.data import InvalidDatasetError, MODEL_FEATURE_COLUMNS, validate_dataset
from sentinel.provenance import PROVENANCE_KEY, metadata_path
from sentinel.v7_audit.audit import CandidateSchedule, audit_candidate, run_audit
from sentinel.v7_audit.__main__ import main, write_new_json
from temporal_fixtures import attest, write_fixture


@pytest.fixture
def frame():
    rows = []
    for repository in ("org/a", "other/b", "third/c"):
        for date in pd.date_range("2021-01-31", "2026-03-31", freq="ME"):
            for label in (0, 1):
                rows.append({"repository": repository, "file_path": f"src/{label}.java",
                             "snapshot_date": date, "defect_next_90_days": label,
                             **dict.fromkeys(MODEL_FEATURE_COLUMNS, 1.0)})
    return attest(validate_dataset(pd.DataFrame(rows)))


def test_candidate_accounting_isolation_and_support(frame):
    result = audit_candidate(frame)
    assert result["status"] == "candidate_only"
    for fold in result["folds"]:
        assert fold["structural_status"] == "passed"
        assert all(fold["checks"].values())
        assert fold["pooled_class_support_passed"]
        assert fold["held_out_repository"] not in fold["source_repositories"]
        development = sum(v["pooled"]["rows"] for v in fold["development"].values())
        assert development + fold["final_evaluation"]["rows"] + fold["excluded_rows"] == len(frame)
        assert "positives" not in fold["final_evaluation"]
        assert all(fold["held_out_repository"] not in v["repositories"] for v in fold["development"].values())
        # Identical relative paths in different repositories are distinct files.
        assert fold["development"]["model_fitting"]["pooled"]["unique_files"] == 4


@pytest.mark.parametrize("boundary,role", [
    ("2022-06-30", "model_fitting"), ("2023-09-30", "model_selection"),
    ("2024-09-30", "calibration_fitting"), ("2025-12-31", "policy_selection"),
])
def test_exact_90_days_excluded_91_days_retained(frame, boundary, role):
    source = frame.iloc[:2].copy()
    extra = []
    for days in (90, 91):
        block = source.assign(snapshot_date=pd.Timestamp(boundary) - pd.Timedelta(days=days),
                              file_path=lambda f: f.file_path + f"-{days}")
        extra.append(block)
    changed = attest(validate_dataset(pd.concat([frame, *extra], ignore_index=True)))
    before = audit_candidate(frame)["folds"][1]
    after = audit_candidate(changed)["folds"][1]
    assert after["development"][role]["pooled"]["rows"] == before["development"][role]["pooled"]["rows"] + 2
    excluded = [r for r in after["exclusions"] if r["reason"] == "immature_" + role + "_label"]
    before_excluded = sum(r["rows"] for r in before["exclusions"] if r["reason"] == "immature_" + role + "_label")
    assert sum(r["rows"] for r in excluded) == before_excluded + 2


def test_leap_day_exact_horizon_and_coverage(frame):
    schedule = CandidateSchedule(model_selection_start="2024-05-29", calibration_start="2024-09-30", policy_start="2025-03-31")
    added = frame.iloc[:2].assign(snapshot_date=pd.Timestamp("2024-02-29"), file_path="leap.java")
    added["file_path"] = ["leap0.java", "leap1.java"]
    baseline = audit_candidate(frame, schedule)["folds"][1]
    result = audit_candidate(attest(validate_dataset(pd.concat([frame, added], ignore_index=True))), schedule)["folds"][1]
    assert result["development"]["model_fitting"] == baseline["development"]["model_fitting"]
    # March 31 + exactly 90 days is June 29, inclusive for observation coverage.
    covered = audit_candidate(attest(frame, "2026-06-29"))
    assert all(f["checks"]["requested_common_test_coverage"] for f in covered["folds"])
    short = audit_candidate(attest(frame, "2026-06-28"))
    assert all(f["structural_status"] == "failed" for f in short["folds"])
    assert all(any(x["reason"] == "incomplete_test_label" for x in f["exclusions"]) for f in short["folds"])


def test_test_labels_cannot_change_any_candidate_output(frame):
    changed = frame.copy()
    final = changed.snapshot_date.ge("2025-12-31")
    changed.loc[final, "defect_next_90_days"] = 1 - changed.loc[final, "defect_next_90_days"]
    assert audit_candidate(frame) == audit_candidate(attest(changed))


def test_held_out_history_labels_cannot_change_its_fold(frame):
    changed = frame.copy()
    changed.loc[changed.repository.eq("org/a"), "defect_next_90_days"] = 0
    before = next(f for f in audit_candidate(frame)["folds"] if f["held_out_repository"] == "org/a")
    after = next(f for f in audit_candidate(attest(changed))["folds"] if f["held_out_repository"] == "org/a")
    assert before == after


def test_empty_and_single_class_partitions_are_reported_without_rescue(frame):
    changed = frame.loc[~frame.snapshot_date.between("2023-09-30", "2024-09-29")].copy()
    changed.loc[changed.snapshot_date.ge("2024-09-30"), "defect_next_90_days"] = 0
    result = audit_candidate(attest(changed))
    for fold in result["folds"]:
        assert fold["structural_status"] == "failed"
        assert "empty_development_partition" in fold["development"]["calibration_fitting"]["issues"]
        assert not fold["development"]["policy_selection"]["pooled_two_class_gate_passed"]
    assert result["schedule"] == audit_candidate(frame)["schedule"]


def test_determinism_and_no_mutation(frame):
    before = frame.copy(deep=True)
    shuffled = attest(frame.sample(frac=1, random_state=3).reset_index(drop=True))
    assert audit_candidate(frame) == audit_candidate(shuffled)
    pd.testing.assert_frame_equal(frame, before)


def test_duplicate_identities_fail_closed_with_counts(frame, tmp_path):
    path = tmp_path / "a_features.csv"
    write_fixture(frame, path)
    raw = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    raw.to_csv(path, index=False)
    result = run_audit([path], tmp_path)
    assert result["input_status"] == "invalid" and result["candidate"] is None
    assert result["datasets"][0]["duplicate_identity_rows"] == 2
    assert result["datasets"][0]["duplicate_identity_excess"] == 1


@pytest.mark.parametrize("mutation", ["csv", "horizon", "missing", "frame_hash"])
def test_provenance_failures(frame, tmp_path, mutation):
    path = tmp_path / "a_features.csv"
    write_fixture(frame, path)
    if mutation == "csv":
        path.write_text(path.read_text().replace("1.0", "2.0", 1))
    elif mutation == "missing":
        metadata_path(path).unlink()
    else:
        metadata = json.loads(metadata_path(path).read_text())
        if mutation == "horizon":
            metadata["horizon_days"] = 91
        else:
            metadata["repositories"]["org/a"]["feature_frame_sha256"] = "0" * 64
        metadata_path(path).write_text(json.dumps(metadata))
    result = run_audit([path], tmp_path)
    assert result["input_status"] == "invalid"
    assert result["candidate"] is None
    assert "error" in result["datasets"][0]


def test_cross_input_duplicates_and_source_hash_mismatch(frame, tmp_path):
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    write_fixture(frame, first)
    write_fixture(frame, second)
    assert run_audit([first, second], tmp_path)["input_status"] == "invalid"
    (tmp_path / "a_java_history.csv").write_text("wrong source")
    assert "Source history hash mismatch" in run_audit([first], tmp_path)["datasets"][0]["error"]


def test_audit_cli_no_fitting_and_exclusive_output(frame, tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    def forbidden(*args, **kwargs):
        raise AssertionError("audit attempted ML fitting")
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden)
    path, output = tmp_path / "input.csv", tmp_path / "audit.json"
    write_fixture(frame, path)
    assert main(["audit", "--root", str(tmp_path), "--features", str(path), "--output", str(output)]) == 0
    saved = output.read_bytes()
    assert json.loads(saved)["input_status"] == "verified"
    with pytest.raises(FileExistsError):
        write_new_json(output, {})
    assert output.read_bytes() == saved
    other = tmp_path / "repeat.json"
    main(["audit", "--root", str(tmp_path), "--features", str(path), "--output", str(other)])
    assert other.read_bytes() == saved


def test_bad_schedule_and_support_configuration(frame):
    with pytest.raises(InvalidDatasetError):
        audit_candidate(frame, CandidateSchedule(calibration_start="2021-01-01"))
    with pytest.raises(ValueError):
        audit_candidate(frame, minimum_class_count=0)
    metadata = deepcopy(frame.attrs[PROVENANCE_KEY])
    metadata["horizon_days"] = 89
    frame.attrs[PROVENANCE_KEY] = metadata
    with pytest.raises(InvalidDatasetError):
        audit_candidate(frame)


def test_baseline_reports_missing_evidence_without_fabricating(tmp_path, monkeypatch):
    from sentinel.v7_audit import baseline
    monkeypatch.setattr(baseline, "git", lambda root, *args: "" if args[0] == "ls-tree" else "a" * 40)
    result = baseline.baseline_manifest(tmp_path)
    assert all(v["status"] == "missing_locally" for v in result["evidence"].values())
    assert not any("retained_content" in v for v in result["evidence"].values())
    assert result["saved_model_files"] == []


def test_preservation_detects_tampering_missing_files_and_tag_change(tmp_path, monkeypatch):
    from sentinel.v7_audit import baseline
    from sentinel.provenance import sha256_file
    source = tmp_path / "source.py"
    source.write_text("original")
    reference = {"files": {"source.py": sha256_file(source)}, "tag_object": "tag",
                 "tag_commit": "commit", "main_commit": "commit"}
    monkeypatch.setattr(baseline, "git", lambda root, *args: "tag" if args[-1] == "v0.6.0" else "commit")
    assert baseline.preservation_report(tmp_path, reference)["status"] == "passed"
    source.write_text("changed")
    assert baseline.preservation_report(tmp_path, reference)["changed"] == ["source.py"]
    source.unlink()
    assert baseline.preservation_report(tmp_path, reference)["missing"] == ["source.py"]
    monkeypatch.setattr(baseline, "git", lambda *args: "different")
    assert not baseline.preservation_report(tmp_path, reference)["checks"]["tag_object_unchanged"]


def test_empty_input_cli_explicit_failure(tmp_path):
    output = tmp_path / "missing-inputs.json"
    assert main(["audit", "--root", str(tmp_path), "--output", str(output)]) == 2
    assert json.loads(output.read_text())["candidate"] is None
