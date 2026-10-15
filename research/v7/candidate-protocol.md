# Candidate experimental protocol — not frozen

This document describes requirements for Phase 2 design. Phase 1 only audits
eligibility; it does not fit models or implement calibration/decision policies.

| Role | Start | Required outcome availability |
|---|---|---|
| Model fitting | Earliest usable history | `t + 90 days < 2022-06-30` |
| Model selection | 2022-06-30 | `t + 90 days < 2023-09-30` |
| Calibration fitting | 2023-09-30 | `t + 90 days < 2024-09-30` |
| Policy selection | 2024-09-30 | `t + 90 days < 2025-12-31` |
| Final historical evaluation | 2025-12-31–2026-03-31 | `t + 90 days <= repository observation endpoint` |

All dates are UTC days and whole repository/date groups stay together. The
horizon is 90 days, not three calendar months. Equality with the next role start
is purged. Coverage of the requested test end is checked against the earliest
endpoint across all inputs, matching the conservative V6 common-coverage rule.
Missing coverage causes a failed diagnostic; no date is silently moved.

Each outer LOPO fold excludes every target row from all four development roles.
Target pre-test history, source rows at/after test start, immature labels,
out-of-range target dates and incomplete labels are counted separately.
Fold identity fingerprints exclude labels. Development summaries use only source
labels; candidate test summaries contain identities/counts/coverage, not labels
or performance. Dataset inventory label distributions are descriptive only.

The audit reports empty roles, missing source repositories, per-repository
single-class support and pooled support. Its default minimum of one instance of
each class is only a necessary computational check. It does not establish power,
calibration stability or independent defect-event counts.

## Later experiment requirements

1. Fit the existing preprocessing and baseline configurations on source training.
2. Select a base model using source model-selection evidence; freeze the model.
3. Fit a prespecified calibration family using independent source calibration
   scores with natural prevalence; retain a no-calibration reference.
4. Select policy parameters using source policy evidence and declared budgets.
5. Lock models, calibrators, policy parameters, metrics and budgets before final
   target evaluation; target labels may then support reporting only.

Do not refit the base model after calibration without new independent calibration
scores. Do not fit a calibrator on in-sample base-model predictions. If a family
is selected rather than prespecified, it needs nested source-only validation,
not reuse of policy or target-test outcomes. Inner source-project holdouts would
emulate project transfer; with three sources their support must first be audited.

Proposed policy comparisons include fixed/default V6 cutoffs, source-selected
thresholds with declared caps, and top-fraction inspection budgets. Candidate
budget fractions are 1%, 5%, 10%; no policy is implemented or selected here.
Rank within each repository and snapshot, with deterministic identity ties and
an explicit rounding rule. A strictly increasing calibration map leaves top-k
membership unchanged; non-strict maps can create ties.

Proposed evaluation includes Brier, log loss with a declared numerical rule,
reliability bins, ECE sensitivity, PR-AUC/prevalence lift, and precision/recall
under budgets. Report per-project/per-window results, macro summaries, worst
cases, undefined-metric denominators and paired uncertainty. Avoid row-IID
confidence intervals; plan repository/time-block analysis. More repositories
and additional prespecified windows are needed for robust generalization claims.

The published V6 test is a historical comparison, not untouched confirmation.
Additional monthly observations or earlier windows already in the histories are
not automatically new independent evidence. Freeze confirmation cohort/windows
before inspecting their outcomes. Unlabeled target data may be scored and ranked
under a locked policy; target-driven adaptation requires a separately declared
transductive protocol, currently unresolved and outside the source-only default.
