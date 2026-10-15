# Phase 1 verification

Verification date: 2026-10-15. Audit/test environment matches the recorded V6
core versions: Python 3.13.9, pandas 2.3.3, NumPy 2.3.5, scikit-learn 1.7.2,
Matplotlib 3.10.6. Pytest 8.4.2 was used. These observations are not a complete
cross-platform reproducibility guarantee.

## Automated checks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src MPLCONFIGDIR=/private/tmp/sentinel-matplotlib \
  python -B -m pytest -o addopts='' -q -p no:cacheprovider
ruff check --no-cache src/sentinel/v7_audit tests/test_v7_audit.py research/v7/verify_evidence.py
git diff --check
```

Result: **275 passed in 37.92 seconds**: all 254 existing tests plus 21 new
Phase 1 tests. Ruff and whitespace checks pass. New Python files parse without
compilation output; evidence JSON is finite and standards-compliant.

New tests cover all four maturity transitions (day 90 excluded/day 91 retained),
leap-day arithmetic and observation endpoint inclusion, empty/single-class roles,
held-out repository isolation, label-independent candidate output, duplicate
identities, source collisions and hash mismatch, missing/stale/incompatible
sidecars, deterministic row/file order, nonmutation, exact row accounting,
missing baseline evidence, preservation tampering and exclusive output creation.
Audit CLI fitting is forbidden by estimator-method stubs during its test.

The full regression suite's existing synthetic model tests fit temporary fixture
models. No new real-data model fitting, calibration or policy evaluation occurred.

## Real evidence verification

`evidence/verification.json` records an independent check of the generated audit
using Python CSV/date arithmetic rather than the audit's pandas masks. It
compares source role rows/classes, final label-free test membership counts,
exclusion-reason counts and complete row accounting for all four folds, plus
verbatim retained JSON summaries and protected-file hashes.

Result: **105 independent checks passed**; all **357 protected files** are
unchanged, with no missing files or added V6 artifacts. Main, the tag object
and the peeled release commit are unchanged.

```bash
PYTHONPATH=src python research/v7/verify_evidence.py
```

The verifier defaults to inspecting the saved V7 evidence. Supply `--output` with
a new path to retain another verification record. It never overwrites evidence.
The baseline manifest was generated twice to separate destinations and compared
byte-for-byte. Real-data dataset audit generation validates V6's canonical
feature hashes using the existing provenance implementation.

## V6 preservation

The pre-change inventory in `evidence/protected-v6.json` binds 357 original
source/test/document/data/artifact files. Verification checks changed/missing
files, additions under existing V6 artifacts, the annotated tag object and its
peeled commit, and the main commit. New V7 files live outside that inventory.
Original tracked files have no diff. The new development branch is
`v7-research-baseline`; `main` and `v0.6.0` remain unchanged. No commit or push was
performed. The phase status and remaining design decisions are described in
`README.md` and `limitations.md`; successful checks do not certify V7 performance.
