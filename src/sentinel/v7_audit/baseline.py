"""Inventory retained V6 evidence without regenerating experimental results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sentinel.provenance import sha256_file
from sentinel.v7_audit.audit import relative_path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def evidence_file(path: Path, root: Path) -> dict:
    record = {"path": relative_path(path, root), "status": "missing_locally"}
    if path.is_file():
        record.update(status="available", sha256=sha256_file(path), bytes=path.stat().st_size)
    return record


def baseline_manifest(root: Path) -> dict:
    retained = root / "artifacts/v6-phase8-revalidation"
    names = ["protocol.json", "environment.json", "datasets.json", "preflight.json",
             "populations.json", "artifact-verification.json", "history-reconstruction.json",
             "calibration-reproducibility.json", "comparison.json",
             "within/metrics.json", "cross/metrics.json",
             "within/calibration_summary.json", "cross/calibration_summary.json"]
    evidence = {}
    for name in names:
        path = retained / name
        record = evidence_file(path, root)
        if path.is_file():
            record["retained_content"] = json.loads(path.read_text())
        evidence[name] = record
    tagged_paths = git(root, "ls-tree", "-r", "--name-only", "v0.6.0^{}").splitlines()
    paths = {root / p for p in tagged_paths}
    paths.update(p for p in (root / "artifacts").rglob("*") if p.is_file())
    paths.update(root.glob("*.csv"))
    files = [evidence_file(p, root) for p in sorted(paths)]
    # Constructing the existing pipelines records effective configuration only;
    # no fit, predict, artifact loading, or result regeneration occurs.
    from sentinel.ml.models import DEFAULT_RANDOM_STATE, build_model_pipelines
    from sentinel.ml.data import MODEL_FEATURE_COLUMNS
    models = build_model_pipelines(MODEL_FEATURE_COLUMNS, DEFAULT_RANDOM_STATE)
    dependencies = [
        {"dependency": "Phase 8 LOPO fitted estimators", "status": "not_retained_in_phase8_outputs",
         "impact": "Cannot reconstruct source calibration scores from test rankings alone."},
        {"dependency": "Independent source calibration/policy score ledgers", "status": "not_retained_in_phase8_outputs",
         "impact": "Future V7 experiments need new authorized source-only scoring, not target-test reuse."},
        {"dependency": "Exact mining checkout HEAD and extraction manifest", "status": "not_recorded_in_v6_provenance",
         "impact": "Supplied CSV hashes are reproducible locally; upstream extraction completeness is unverified."},
        {"dependency": "Locked install environment", "status": "versions_recorded_but_no_lockfile_in_v6",
         "impact": "Broad pyproject ranges do not guarantee identical retraining across installations."},
        {"dependency": "Original temporary script paths in evaluation-commands.txt", "status": "machine_specific",
         "impact": "Retained scripts/ directory exists; original /private/tmp paths are not portable dependencies."},
        {"dependency": "Histories, feature CSVs, sidecars and experiment outputs in a fresh clone",
         "status": "ignored_not_distributed", "impact": "Numerical reproduction requires separately supplied hash-matching files."},
    ]
    return {
        "schema_version": 1, "release": "v0.6.0", "evidence_scope": "corrected V6 Phase 8 historical studies",
        "git": {"main_commit": git(root, "rev-parse", "main"),
                "tag_object": git(root, "rev-parse", "v0.6.0"),
                "tag_commit": git(root, "rev-parse", "v0.6.0^{}")},
        "evidence": evidence, "file_inventory": files,
        "model_configuration": {
            "basis": "unchanged V6 model factory, effective defaults in audit environment; no fitting",
            "source": evidence_file(root / "src/sentinel/ml/models.py", root),
            "predictors": MODEL_FEATURE_COLUMNS, "random_state": DEFAULT_RANDOM_STATE,
            "estimators": {name: model.named_steps["model"].get_params(deep=False) for name, model in models.items()},
            "preprocessing": {"logistic_regression": "StandardScaler fitted on training only",
                              "dummy_and_random_forest": "historical numeric whitelist passthrough"},
        },
        "selection": {"model": "source validation PR-AUC, then F1, then recall",
                      "classification_threshold": "validation maximum F1",
                      "calibration": "diagnostic only; ten equal-width bins; no fitted calibrator",
                      "retrospective_thresholds": "0 to 1 by 0.05; max F1, highest threshold on ties; never tuning input"},
        "reproducibility_dependencies": dependencies,
        "saved_model_files": [relative_path(p, root) for p in sorted((root / "artifacts").rglob("*.joblib"))],
        "interpretation": "Retained results copied, not regenerated; no performance, novelty or deployment claim.",
    }


def preservation_report(root: Path, reference: dict) -> dict:
    changed, missing = [], []
    for name, digest in reference["files"].items():
        path = root / name
        if not path.is_file():
            missing.append(name)
        elif sha256_file(path) != digest:
            changed.append(name)
    current_artifacts = {relative_path(p, root) for p in (root / "artifacts").rglob("*") if p.is_file()}
    old_artifacts = {p for p in reference["files"] if p.startswith("artifacts/")}
    checks = {"protected_files_unchanged": not changed and not missing,
              "no_added_v6_artifacts": current_artifacts == old_artifacts,
              "tag_object_unchanged": git(root, "rev-parse", "v0.6.0") == reference["tag_object"],
              "tag_commit_unchanged": git(root, "rev-parse", "v0.6.0^{}") == reference["tag_commit"],
              "main_unchanged": git(root, "rev-parse", "main") == reference["main_commit"]}
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "protected_file_count": len(reference["files"]), "changed": changed, "missing": missing,
            "added_artifacts": sorted(current_artifacts - old_artifacts)}
