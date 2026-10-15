# Future experimental data contracts — design only

These are proposed logical schemas, not implemented ledgers, calibration or
policy interfaces. Role names match the candidate audit. Schema versions and
the final protocol remain decisions for Phase 2 design.

## Score ledger

One row per `(experiment_id, outer_fold_id, model_id, role, repository,
file_path, snapshot_date, score_variant)`. Repository/path/date alone identifies
the underlying sample. Store identities as separate fields; do not depend on
ambiguous delimiter concatenation. Paths are repository-relative, normalized and
case-preserving; a renamed path is a different V6 entity unless a future dataset
version explicitly supplies lifecycle linkage.

| Field group | Required fields / constraints |
|---|---|
| Version and experiment | `schema_version`, `experiment_id`, `outer_fold_id`, `protocol_id`, `protocol_sha256`, `held_out_repository` |
| Sample | `repository`, `file_path`, `snapshot_date` (UTC day), `label_window_end` (= date + 90 days), `observation_end`, `dataset_id` |
| Role | `model_fitting`, `model_selection`, `calibration_fitting`, `policy_selection`, `final_evaluation`; role start/end, eligibility rule, exclusion reason when excluded |
| Model | `model_id`, `model_lineage_sha256`, `score_method`, `raw_score`, `score_variant`; raw score finite in [0,1] for current baselines |
| Provenance | source-history hash, feature CSV hash, canonical feature-frame hash, sidecar hash, identity-partition hash, Sentinel/code revision, environment ID |
| Outcome access | `label_reference`, `label_access_scope`; development labels source-only, final labels in separate evaluator-owned table |
| Optional future output | calibrated score and calibrator ID only after a calibrator exists; policy ID and decision only after a policy exists; absent fields are null, never fabricated |

Snapshot features and identity may be needed for scoring, but repository/path/date
must remain metadata outside the eight-feature predictor whitelist. The score
ledger is not a channel to inject new predictors or fit preprocessing on target
data. Base-model fitting scores cannot serve as calibration training data.

## Model lineage record

Record estimator family and complete effective parameters, seed, predictor list,
preprocessing definition and fitting scope, source repository/family identities,
training dates and maturity boundaries, training membership hashes, model artifact
hash, selection evidence/criterion, selection lock time or ordered event record,
code revision and dependency versions. Do not confuse a code commit with a model
identity. Target absence must be verifiable through all parent lineage records.

A calibrator lineage must include its frozen parent model hash, family/parameters,
source calibration membership, label availability, weights, input score method,
fit provenance and any separate family-selection record. Any model refit changes
lineage and invalidates an old calibration association until independently checked.

## Calibration and policy inputs

Calibration fitting accepts paired finite source scores/binary mature labels
from its designated role, independent of base-model fitting and selection.
Preserve natural source prevalence unless a weighting estimand is declared.
Validate two-class support; report source imbalance and dependence. Numerical
sample count alone does not establish sufficient calibration evidence.

Policy selection accepts raw or calibrated source scores from the separate
policy role and mature source labels. Declare budget unit, fractions/caps,
objective, tie/rounding rules and eligible population. Decisions at a snapshot
may use only information available then. Full-evaluation-period hotspot counts
cannot be used as historical online evidence.

If calibration family/weights or policy family is optimized, document a separate
nested source validation layer. Do not silently optimize several stages on one
role. Cross-project pseudo-target experiments must exclude the pseudo-target
from their fitting stage and exclude the true outer target everywhere.

## Test-label isolation and missingness

Final labels join locked predictions only in final evaluation. Candidate schedule
audits use label-free test membership fingerprints. Integrity verification may
hash labels and validate their binary structure; this is not selection access.
Published V6 diagnostics may be inventoried as historical evidence but cannot
select V7 parameters. Cross-fold predictions are not automatically reusable:
their training lineage may contain the current target.

Fail closed for stale hashes, incompatible horizon/schema, unknown lineage,
duplicate sample keys, immature labels or out-of-range/nonfinite scores. Excluded
rows retain reason/count accounting; do not silently clip scores, impute labels,
or replace missing project identities with fictitious projects.

Undefined measurements: JSON `null`, CSV blank, Markdown `N/A`, with reason and
eligible denominator. Empty reliability bins have count 0 and null means.
Zero-positive tests have unavailable recall/PR-AUC/ROC-AUC where undefined;
preserve the V6 legacy zero-division metrics separately when reproducing V6.
Never overwrite the original values to harmonize a future report. Every metric
summary records projects/windows contributing and those unavailable.

All exported JSON is finite and standards-compliant. Sort records by explicit
identity fields and serialize deterministically. Manifest paths are relocatable
where possible; hashes, not local absolute paths, establish content identity.
