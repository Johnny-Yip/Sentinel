# Sentinel V7 — Phase 1: Research Baseline & Dataset Audit

Research direction: **Robust Cross-Project Defect Prediction**. Phase 1 builds
auditable evidence and planning documents, not a new predictor or calibration
method. V6 v0.6.0 remains the stable research baseline.

**Status: Phase 1 complete.** The full suite passes (275 tests), independent
evidence verification passes (105 checks), and all 357 protected V6 files, main
and the release tag remain unchanged. Ready to design Phase 2; no Phase 2
implementation, protocol freeze or experiment has begun.

## Contents

| Document | Purpose |
|---|---|
| [Research questions](research-questions.md) | Questions, estimands and falsifiable hypotheses |
| [V6 baseline](v6-baseline.md) | Retained results, provenance, reusable components and missing dependencies |
| [Dataset audit](dataset-audit.md) | Local inventory and candidate-schedule findings |
| [Candidate protocol](candidate-protocol.md) | Independent role requirements and unresolved design choices |
| [Expansion plan](dataset-expansion.md) | Reproducible admission and reserved confirmation policy |
| [Literature review](literature-review.md) | Verified references, supported findings and applicability limits |
| [Data contracts](data-contracts.md) | Future ledgers, lineage, access rules and undefined values |
| [Limitations and decisions](limitations.md) | Boundaries of evidence and Phase 2 design prerequisites |
| [Verification](verification.md) | Test and preservation evidence |

Machine-readable evidence lives in `evidence/`. No raw datasets, model binaries
or copies of V6 experimental outputs are introduced. The baseline manifest
contains retained JSON summaries and file hashes; it does not regenerate results.

## Reproduce inspection

From the repository root, using the recorded V6 environment and locally supplied
hash-matching inputs:

```bash
PYTHONPATH=src python -m sentinel.v7_audit audit --output /private/tmp/v7-audit-new.json
PYTHONPATH=src python -m sentinel.v7_audit baseline --output /private/tmp/v7-baseline-new.json
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/sentinel-matplotlib python -m pytest -q -p no:cacheprovider
```

Output paths must not exist: exclusive creation protects retained evidence.
The audit defaults to the four Phase 8 full monthly feature CSVs. Override with
`--features PATH ...` and `--root ROOT`. Missing/stale/incompatible provenance,
duplicates and repeated repositories fail closed; invalid inputs produce a JSON
diagnostic and exit status 2. Histories are located by V6 repository basename;
missing histories are explicitly reported, and present mismatched histories fail.
No network access, estimator fitting, prediction or calibration is performed by
the audits. The baseline command constructs V6 model objects solely to inspect
their effective configuration; it never fits or loads a fitted estimator.

The existing regression suite includes synthetic estimator fits in temporary
directories. Running it does not launch new real-data experiments.

Phase 1 is complete when the evidence, focused/regression tests and preservation
checks pass. Phase 2 may then be **designed**, but no Phase 2 implementation or
protocol freeze is authorized by these documents.
