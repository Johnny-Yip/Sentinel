# Sentinel V6 release summary — 0.6.0

V6 Phases 1–8 are implemented on `v6-explainable-risk`, through implementation
commit `32d9253328e721cd69ec2fc94d7ecaa8f076bd39`. This release adds explainable
file-snapshot risk intelligence to the existing research pipeline and corrects
outcome-window leakage in historical temporal evaluation. Package and runtime
versions remain **0.6.0**. Release preparation changes documentation only.

## Completed scope

| Phase | Achievement |
|---|---|
| 1 | Bounded risk scores, deterministic ranking, and high-risk summaries |
| 2 | Individual linear/tree-path explanations and deterministic fallback attribution |
| 3 | Stable feature-based inspection guidance with explicit unsupported-signal handling |
| 4 | Project concentration, recurring hotspots, descriptive trends, and priority queues |
| 5 | Unified developer risk briefs and project-local action tiers |
| 6 | Diagnostic explanation stability, evidence coverage, action specificity, and risk alignment |
| 7 | Per-project Brier/ECE, reliability bins, and retrospective threshold tradeoffs from retained predictions |
| 8 | Hash-bound source/feature provenance, mature-label purging, shared-calendar LOPO, pre-fit audits, and real-data revalidation |

Mining and monthly features remain separate from the centralized dummy,
class-weighted Logistic Regression, and Random Forest baselines. Training uses
only eight historical predictors; preprocessing fits on training, model choice
uses validation PR-AUC (then F1/recall ties), and classification thresholds use
validation F1. All choices are locked before test scoring. The unified report
adapters add 21 artifacts without a second ML implementation. Downstream risk
summaries, explanations, guidance, diagnostics, and calibration do not alter
selection. There is no application service or UI.

## Corrected protocol and retained results

For a snapshot on UTC day `t`, features use history through that day and the
label observes `(t, t + 90 days]`. Training labels end strictly before validation
starts; validation labels end strictly before testing starts. Test label windows
must end on or before each repository's declared observation endpoint.
Repository/date groups stay together. Missing/stale provenance, empty required
partitions, or incompatible boundaries fail closed; boundaries are not moved to
improve class balance or performance. LOPO excludes the held-out repository from
both fitting and selection and uses one shared calendar schedule across folds.

The retained study uses seed 42, the unchanged baseline configurations, ten
calibration bins, and retrospective threshold candidates `0, 0.05, …, 1`.

- **Within-project:** full monthly commons-lang, validation `2019-03-31`, test
  start `2022-10-31`, coverage-limited test end `2026-06-10` (last available
  snapshot `2026-05-31`). Retained train/validation/test: **152,016 / 51,942 /
  62,008**. Purged **3,583 training** and **2,720 validation** snapshots.
- **Cross-project:** quarter-end snapshots from `2023-06-30` through
  `2026-03-31`; shared validation `2025-09-30`; shared test `2025-12-31` through
  `2026-03-31`. Four LOPO folds start from the same **76,345** input rows.
  Exclusions are held-out pre-test history and source rows at/after test start;
  quarter spacing already satisfies label maturity at these boundaries.

All five evaluations selected Logistic Regression. Classification metrics below
use the locked validation-selected threshold, not the retrospective recommendation.

| Evaluation / project | Test rows | Positives | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Within / commons-lang | 62,008 | 1,281 | 0.208494 | 0.210773 | 0.209627 | 0.889081 | 0.163370 | 0.133170 | 0.194599 |
| LOPO / commons-collections | 5,282 | 44 | 0.042952 | 0.886364 | 0.081933 | 0.929028 | 0.092859 | 0.131386 | 0.191786 |
| LOPO / commons-io | 1,931 | 39 | 0.428571 | 0.230769 | 0.300000 | 0.878571 | 0.268971 | 0.356487 | 0.469541 |
| LOPO / commons-lang | 2,897 | 87 | 0.124521 | 0.747126 | 0.213465 | 0.890578 | 0.234027 | 0.188381 | 0.251331 |
| LOPO / junit4 | 2,810 | 0 | 0.000000 | 0.000000 | 0.000000 | N/A | N/A | 0.009865 | 0.033291 |

JUnit4's zero precision/recall/F1 follow the existing classifier convention and
provide no evidence of positive-case detection. ROC-AUC, PR-AUC, and the
retrospective recommendation are unavailable. Calibration threshold tables keep
zero-denominator rates unavailable (`null`/blank/`N/A`).

| Evaluation / project | Validation-selected threshold | Retrospective max-F1 candidate |
|---|---:|---:|
| Within / commons-lang | 0.917288 | 0.85 |
| LOPO / commons-collections | 0.683407 | 0.85 |
| LOPO / commons-io | 0.944542 | 0.95 |
| LOPO / commons-lang | 0.735937 | 0.90 |
| LOPO / junit4 | 0.745837 | N/A |

These thresholds are project-sensitive. On matched test identities, Phase 7 to
Phase 8 within-project PR-AUC changed `0.162103 → 0.163370`, F1 changed
`0.208921 → 0.209627`, and Brier/ECE slightly worsened. LOPO scores and rankings
were unchanged on matched identities, but stricter validation changed thresholds:
collections F1 `0.176166 → 0.081933`, io `0.293333 → 0.300000`, and lang
`0.265560 → 0.213465`. Every comparison retained all top 20 identities; within
score rank correlation was `0.999895`. Stable rankings do not establish stable
operating thresholds. Legacy Phase 7 results remain historically unverified;
these comparisons are descriptive, not estimates of leakage inflation.

## Evidence and final quality assurance

Release preparation on **2026-10-14** verified the following locally:

- Full regression suite: **254 passed in 33.63 seconds** with
  `PYTHONPATH=src MPLCONFIGDIR=/private/tmp/sentinel-matplotlib python -m pytest`.
- All five partitions/maturity audits recomputed from retained CSVs and sidecars;
  partition hashes, counts, exclusions, and boundaries match the saved audits.
- All four source-history hashes, observation endpoints, feature CSV hashes,
  canonical frame hashes, and quarterly subsets verified against retained inputs.
- Eligible test identities and labels match the rankings; classification metrics,
  confusion matrices, inclusive decision rules, and deterministic ordering agree.
- Both evaluations contain all **21 artifacts**, with finite standards-compliant
  JSON. All **six calibration files** reproduce byte-for-byte in separate
  temporary output directories.
- A fresh sampled reconstruction of **404 snapshots** matches the eight features
  and labels. It reuses the history normalization and bug-fix heuristic; it is
  an arithmetic/window check, not independent defect-label ground truth.
- Python compilation, library imports, all five console-entry targets, CLI example
  parsing, Ruff, and whitespace checks pass. The quarterly preparation example
  was also executed for commons-io, reproducing its input CSV byte-for-byte. Source/tests and package version are
  unchanged; the release file set contains only documentation and no detected
  secrets, datasets, model binaries, or temporary artifacts.

The original Phase 8 record additionally documents complete feature regeneration
with all four CSVs byte-identical and checks of explanations and action artifacts.
Full real-data model fitting was performed once per mode in that revalidation;
it was **not repeated during final release preparation**. Fresh checks establish
retained-input integrity and deterministic artifact-only calibration, not
byte-identical end-to-end retraining across platforms or dependency versions.

Evidence is retained locally under `artifacts/v6-phase8-revalidation/`:
`REVALIDATION_REPORT.md`, `protocol.json`, `environment.json`, `datasets.json`,
`preflight.json`, `populations.json`, `history-reconstruction.json`,
`artifact-verification.json`, `comparison.json`, `phase7-phase8-metrics.csv`,
`calibration-reproducibility.json`, and the `within/` and `cross/` outputs.
These ignored files are not included in a fresh Git clone. The tables here are
rounded summaries of those retained results, not newly fitted experiments.

## Reproduction requirements

Exact results require the same mining-history files, derived sidecars, and
recorded environment: Python **3.13.9**, pandas **2.3.3**, NumPy **2.3.5**,
scikit-learn **1.7.2**, Matplotlib **3.10.6**. The package's supported dependency
ranges are broader than this recorded environment; identical outputs across
those ranges have not been established. The miner clones current HEAD, so a
fresh mining run may change coverage, rows, and metrics. Provenance binds the
supplied history; upstream extraction completeness has not been certified.

The retained source-history SHA-256 identifiers are:

| Repository | History rows | Observation endpoint | Source-history SHA-256 |
|---|---:|---|---|
| apache/commons-collections | 30,291 | 2026-09-07 | `689e7e41606702536e6d66056940d7a428ea09a47d91717d1c4719b5e5512dac` |
| apache/commons-io | 15,561 | 2026-08-28 | `3b8736f4804790a489a24387d7e9de5d0d2c1131408c1ba655add3588aea67c5` |
| apache/commons-lang | 22,906 | 2026-09-08 | `11b6bed4b4c8d49779dbd0661f373872b621e7fb4b4af780b6b91df91068bfaa` |
| junit-team/junit4 | 10,472 | 2026-08-13 | `cc8ff158c0ee7686d459579add87b2ea4d6d1891b275450fc41d5221d0867da2` |

Generate full monthly CSVs and adjacent sidecars with `sentinel-features` as
shown in the README. To reproduce the bounded quarterly input population from
those verified full datasets in `data/`, retain the same history coverage and
recompute subset hashes rather than copying a full-dataset sidecar:

```python
from copy import deepcopy
from pathlib import Path
from sentinel.ml.data import load_dataset
from sentinel.provenance import PROVENANCE_KEY, feature_digest, save_feature_metadata

for name in ("commons-collections", "commons-io", "commons-lang", "junit4"):
    data = load_dataset(Path("data") / f"{name}_features.csv")
    subset = data.loc[
        data.snapshot_date.between("2023-06-30", "2026-03-31")
        & data.snapshot_date.dt.month.isin([3, 6, 9, 12])
    ].copy()
    # load_dataset verifies the full source first; preserve its coverage/hash.
    metadata = deepcopy(data.attrs[PROVENANCE_KEY])
    subset.attrs[PROVENANCE_KEY] = metadata
    for repository, rows in subset.groupby("repository"):
        metadata["repositories"][repository]["row_count"] = len(rows)
        metadata["repositories"][repository]["feature_frame_sha256"] = feature_digest(rows)
    destination = Path("data/quarterly") / f"{name}_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(destination, index=False, date_format="%Y-%m-%d")
    save_feature_metadata(subset, destination)
```

With matching inputs, use new output directories to preserve retained results:

```bash
sentinel-evaluate within-project data/commons-lang_features.csv \
  --validation-start 2019-03-31 --test-start 2022-10-31 \
  --random-state 42 --output-dir reports/v6-within
sentinel-evaluate cross-project \
  data/quarterly/commons-collections_features.csv \
  data/quarterly/commons-io_features.csv \
  data/quarterly/commons-lang_features.csv data/quarterly/junit4_features.csv \
  --validation-start 2025-09-30 --test-start 2025-12-31 --test-end 2026-03-31 \
  --random-state 42 --output-dir reports/v6-cross
sentinel-evaluate calibrate reports/v6-cross/risk_ranking.csv \
  --evaluation-mode cross_project --output-dir reports/v6-cross-calibration-check
```

## Remaining limitations and release decision

No correctness blocker was found in this final audit. V6 is ready for review,
merge into `main`, and an annotated `v0.6.0` tag as a **research software release**.
No commit, push, merge, tag, or release publication was performed during this
preparation. Approval of the local documentation changes is still required.

Unresolved empirical limitations remain: heuristic bug-fix labels are not
independently verified defects or SZZ ground truth; deleted/renamed paths remain
in snapshots; repeated files and overlapping test windows reduce independence;
the four-project quarterly transfer study is small and does not cover full
monthly LOPO histories. Low precision, missed positives, poor calibration
(particularly commons-io), and threshold sensitivity preclude deployment-ready
claims. JUnit4 lacks positive-case evaluation evidence. The monthly within and
quarterly cross studies cannot establish a matched same-vs-cross advantage.

Explanations and actions describe model associations and deterministic inspection
heuristics, with no causal validation or human usefulness/defect-prevention study.
ECE depends on binning; no uncertainty intervals, formal calibration tests, live
prospective trial, recalibration fit, or universal operational threshold is
provided. Retrospective max-F1 recommendations never replace the validation
threshold. The retained evidence is locally auditable, but exact public numerical
reproduction requires access to the same histories and environment.
