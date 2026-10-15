# Preserved V6 baseline

Authoritative release: [`V6_RELEASE_SUMMARY.md`](../../V6_RELEASE_SUMMARY.md)
and [`README.md`](../../README.md). Corrected Phase 8 studies are the baseline;
pre-Phase-8 evidence remains historical/unverified.

`evidence/v6-baseline.json` inventories actual retained evidence, file sizes and
SHA-256 values. It records all-model metrics, fold choices, calibration summaries,
source/feature identities, observation endpoints, experiment schedules and
environment directly from Phase 8 JSON. `evidence/protected-v6.json` captures
the pre-change hashes of 357 V6 source/test/document/data/artifact files plus
the tag object, tag commit and main commit.

## Release identity and protocols

At Phase 1 start the working tree was clean. `main` and the peeled `v0.6.0` tag
both resolved to `73a0812bb8032625d0079f89dad7e99b4b7d60f5` (Merge Sentinel V6).
The annotated tag object was `8088dce07526d74cfab30cb70cd19cb795c2052a`.
Development uses `v7-research-baseline`. No tag, commit, push or release operation
is part of Phase 1.

| Study | Retained schedule |
|---|---|
| Within commons-lang, monthly | Validation starts 2019-03-31; test starts 2022-10-31; coverage-limited end 2026-06-10, last snapshot 2026-05-31 |
| Four-project LOPO, quarterly | Input 2023-06-30–2026-03-31; validation 2025-09-30; test 2025-12-31–2026-03-31 |

V6 uses exactly eight historical predictors, train-only preprocessing, validation
PR-AUC model selection (F1/recall ties), and validation maximum-F1 classification
thresholds. Labels mature strictly before the next partition begins. LOPO
excludes the target repository from fitting and selection on a common calendar.

## Selected-model reference results

All five evaluations selected Logistic Regression. Full precision and every
baseline model remain in the manifest; these rounded values are not new results.

| Study / project | Test rows / positives | PR-AUC | F1 | Brier | ECE | Selected threshold | Retrospective max-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Within / lang | 62,008 / 1,281 | .163370 | .209627 | .133170 | .194599 | .917288 | .85 |
| LOPO / collections | 5,282 / 44 | .092859 | .081933 | .131386 | .191786 | .683407 | .85 |
| LOPO / io | 1,931 / 39 | .268971 | .300000 | .356487 | .469541 | .944542 | .95 |
| LOPO / lang | 2,897 / 87 | .234027 | .213465 | .188381 | .251331 | .735937 | .90 |
| LOPO / junit4 | 2,810 / 0 | N/A | 0* | .009865 | .033291 | .745837 | N/A |

*Legacy zero-division convention; no evidence of positive detection. Retrospective
recommendations use test outcomes and must never become V7 tuning inputs.
V6 calibration uses ten equal-width bins and Brier/ECE diagnostics; no calibrator
was fitted. Monthly within and quarterly cross populations are not matched.

## Reuse and reproducibility

Reuse `features.py`, `provenance.py`, `ml/data.py`, `cross_project/data.py`,
the eight-feature whitelist, unchanged model factory and metric conventions.
The V6 splitters remain the maturity-rule reference. The V7 audit separately
describes five candidate roles and does not return a new training split API.
Reporting, deterministic rankings and per-project shift summaries are useful
later. Full-period recurrence-based queues are retrospective summaries, not
automatically valid sequential policies.

Recorded environment: Python 3.13.9, pandas 2.3.3, NumPy 2.3.5,
scikit-learn 1.7.2, Matplotlib 3.10.6. Effective classifier parameters are
inspected from unchanged V6 code in this environment; inherited defaults are
identified as reconstructed configuration rather than originally saved metadata.

Available locally: all four histories, full monthly CSVs/sidecars, quarterly
CSV/sidecars, both 21-file evaluation directories and audit records. Ignored
data/artifacts are not distributed in a fresh clone. Missing dependencies are
listed explicitly in the manifest: separate source calibration/policy score
ledgers, Phase 8 LOPO fitted estimators, mining checkout/extraction completeness
records and a fully locked installation. Existing saved model files are
inventoried without loading them and are not substitutes for those estimators.
