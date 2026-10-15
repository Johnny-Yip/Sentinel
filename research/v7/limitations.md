# Known limitations and unresolved decisions

## Evidence limitations

- Four repositories, three from one organization, cannot support broad claims
  about Java software. Additional rows/windows do not substitute for projects.
- Snapshot positives overlap in time and by file/fix event. Effective sample
  size and independent bug-fix event counts are unknown.
- Commit-message labels have unmeasured precision/recall against verified defects.
  Carried-forward deleted/renamed paths affect populations and apparent budgets.
- Observation endpoints use the last supplied Java change. Hash verification
  does not certify upstream extraction completeness or precise active-file state.
- JUnit4 has sparse calibration-period positives and no policy-period positives
  under the candidate schedule. Pooled source roles remain two-class, but a
  repository-specific fit/selection rule may be unsupported.
- V6 test outcomes are known. The proposed historical comparison is exploratory;
  preserved V6 and candidate monthly test populations are not identical.
- Candidate fitting ends in March 2022, well before the late-2025 test start.
  This long model-age gap may confound transfer with temporal drift. Any later
  refitting design needs fresh independent calibration evidence, not reuse of a
  calibrator tied to the earlier model.
- Retained V6 calibration is diagnostic only. No source calibration/policy score
  ledgers or Phase 8 LOPO fitted models are available for reuse.

## Decisions before a Phase 2 protocol freeze

1. Confirm repository admission, independent families and reserved cohort; decide
   whether four-project pilot evidence is sufficient for the next design step.
2. Freeze calendar schedules, test cadence, additional evaluation windows and
   mature-label exclusion rules without inspecting target performance.
3. Prespecify calibrator families or design nested source-only family selection.
   Audit its support; current five-role counts do not verify nested folds.
4. Choose pooled/source-balanced weighting and the target estimand. Sources
   differ greatly in row counts and prevalence; target prevalence is unavailable.
5. Define model selection and policy family/parameter selection separation;
   freeze base models before compatible calibration scores are produced.
6. Set minimum evidence requirements using class counts, positive dates and
   independent events where available; the audit's one-per-class gate is not
   statistical sufficiency. Specify fail/skip behavior in advance.
7. Define inspection unit, budgets, ties/rounding, cost assumptions and metric
   hierarchy. Current data cannot support measured review-time claims.
8. Specify paired project/time-block uncertainty, multiplicity handling and
   meaningful effect reporting. Four-project population inference is weak.
9. Decide whether any unlabeled-target adaptation is in scope. Source-only
   default permits locked scoring/ranking but no target-driven method fitting.
10. Review literature beyond this bounded seed review before any novelty claim.

## Scope decision

Phase 1 establishes a reproducible audit foundation. It can be complete while
these research decisions remain open. The next authorized activity may be
**Phase 2 design**, not an automatic experiment run or implementation. No V7
performance improvement, scientific novelty, universal calibration guarantee or
deployment readiness is established. No new repository has been acquired.
