"""Independently count real-data role populations and verify preserved V6 files.

Run from the repository root with PYTHONPATH=src. This verifier uses CSV and
datetime, not the candidate audit's pandas masks. It does not read test labels
for candidate counts or fit any estimators.
"""

import argparse
from collections import Counter
import csv
from datetime import date, timedelta
import json
from pathlib import Path

from sentinel.v7_audit.baseline import preservation_report
from sentinel.v7_audit.__main__ import write_new_json


def verify(root):
    evidence = root / "research/v7/evidence"
    audit = json.loads((evidence / "dataset-audit.json").read_text())
    baseline = json.loads((evidence / "v6-baseline.json").read_text())
    reference = json.loads((evidence / "protected-v6.json").read_text())
    candidate = audit["candidate"]
    boundaries = [date.fromisoformat(candidate["schedule"][k]) for k in (
        "model_selection_start", "calibration_start", "policy_start", "test_start", "test_end")]
    roles = ("model_fitting", "model_selection", "calibration_fitting", "policy_selection")
    source_counts, reasons, tests, totals = Counter(), Counter(), Counter(), Counter()
    for dataset in audit["datasets"]:
        with (root / dataset["path"]).open(newline="") as handle:
            for row in csv.DictReader(handle):
                repository = row["repository"]
                t = date.fromisoformat(row["snapshot_date"])
                maturity = t + timedelta(days=90)
                observed = date.fromisoformat(dataset["repositories"][repository]["observation_end"])
                totals[repository] += 1
                if t >= boundaries[3]:
                    if maturity > observed:
                        tests[repository, "incomplete_test_label"] += 1
                    elif t > boundaries[4]:
                        tests[repository, "after_test_end"] += 1
                    else:
                        tests[repository, "eligible"] += 1
                    continue  # No candidate test labels read below this point.
                for role, next_start in zip(roles, boundaries[:4], strict=True):
                    if t < next_start:
                        break
                if maturity > observed:
                    reasons[repository, "incomplete_development_label"] += 1
                elif maturity >= next_start:
                    reasons[repository, "immature_" + role + "_label"] += 1
                else:
                    source_counts[repository, role, "rows"] += 1
                    source_counts[repository, role, "positives"] += int(row["defect_next_90_days"])
    checks = {}
    for fold in candidate["folds"]:
        target = fold["held_out_repository"]
        sources = fold["source_repositories"]
        for role in roles:
            for key in ("rows", "positives"):
                expected = sum(source_counts[r, role, key] for r in sources)
                checks[f"{target}:{role}:{key}"] = expected == fold["development"][role]["pooled"][key]
            for r in sources:
                actual = fold["development"][role]["repositories"][r]
                checks[f"{target}:{role}:{r}"] = all(
                    actual[k] == source_counts[r, role, k] for k in ("rows", "positives"))
        expected_exclusions = Counter()
        for r in sources:
            expected_exclusions[r, "source_at_or_after_test_start"] = sum(v for (repo, why), v in tests.items() if repo == r)
            expected_exclusions.update({(repo, why): v for (repo, why), v in reasons.items() if repo == r})
        expected_exclusions[target, "held_out_history"] = totals[target] - sum(v for (r, why), v in tests.items() if r == target)
        expected_exclusions.update({(r, why): v for (r, why), v in tests.items() if r == target and why != "eligible"})
        actual_exclusions = Counter({(r["repository"], r["reason"]): r["rows"] for r in fold["exclusions"]})
        checks[f"{target}:exclusion_reasons"] = +expected_exclusions == actual_exclusions
        checks[f"{target}:test_rows"] = tests[target, "eligible"] == fold["final_evaluation"]["rows"]
        checks[f"{target}:row_accounting"] = sum(source_counts[r, role, "rows"] for r in sources for role in roles) + tests[target, "eligible"] + sum(expected_exclusions.values()) == sum(totals.values())
    # Confirm retained metric/calibration JSON values were copied verbatim.
    for name, record in baseline["evidence"].items():
        if "retained_content" in record:
            checks["retained_json:" + name] = json.loads((root / record["path"]).read_text()) == record["retained_content"]
    preserved = preservation_report(root, reference)
    return {"status": "passed" if all(checks.values()) and preserved["status"] == "passed" else "failed",
            "independent_checks": checks, "preservation": preserved,
            "verification_scope": "Independent CSV/date population accounting and retained JSON/hash checks; no model fitting or performance evaluation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    if args.output:
        write_new_json(args.output, result)
    print(json.dumps({"status": result["status"], "independent_checks": len(result["independent_checks"]),
                      "protected_files": result["preservation"]["protected_file_count"]}))
    raise SystemExit(0 if result["status"] == "passed" else 2)
