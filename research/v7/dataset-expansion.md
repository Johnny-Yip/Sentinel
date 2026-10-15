# Dataset expansion and admission policy

Goal: approximately 10–15 diverse Java repositories, conditional on acquisition
cost and research needs. This is a planning target, not a power guarantee. No new
repository is downloaded, mined or empirically admitted during Phase 1.

Current cohort: Apache commons-collections, commons-io, commons-lang and
junit-team/junit4. Three share an organization and library ecosystem. JUnit4
adds another organization but does not supply broad application-domain diversity.
Repository names and organizations alone do not establish independence.

## Reproducible admission procedure

1. Before looking at predictions or target outcomes, create a candidate registry
   from a declared public source/query with retrieval date and an ordered list.
   Record URL, canonical slug, organization, domain and rationale; domain labels
   need a documented source. Preserve rejected candidates and reasons.
2. Screen Java activity, repository accessibility/license and historical coverage.
   Require enough common UTC history for all candidate development roles and a
   fully observed 90-day evaluation horizon. Do not move boundaries per project
   to manufacture positives or improve results.
3. Record immutable extraction HEAD, branch/history scope, extraction tool/version,
   extraction date, raw-history hash, endpoint and feature-generation provenance.
   Missing evidence blocks confirmed admission rather than being inferred.
4. Check forks, shared commit ancestry, mirrors, renamed repositories and copied
   subsystems. Group related repositories as a family; keep a family in the same
   development/confirmation cohort and avoid claiming independent folds for it.
5. Validate schema, dates, duplicate identities, missing/nonfinite features,
   observation coverage and development class support using the audit tool.
   Label heuristics must remain compatible with the frozen V6 baseline. Data
   quality changes need a separately documented dataset version.
6. Admit on declared quality/coverage criteria. Never use favorable held-out
   performance, target-test positive counts or retrospective thresholds. Empty
   source roles block the affected experiment. Source single-class warnings
   require a prespecified policy; zero-positive final tests remain in reporting.

Registry fields: candidate ID, URL/slug, retrieval source/date, organization,
domain/source, family ID/evidence, Java-activity evidence, first/last history,
coverage feasibility, lifecycle/extraction caveats, admission status/reason,
cohort assignment and registry version/hash. Unknown values stay unknown.

## Diversity and confirmation

Seek multiple organizations and domains such as developer tools, data systems,
services and applications, rather than mostly additional Apache Commons libraries.
As planning goals, seek at least four organizations and three domains and avoid
one organization dominating new admissions. These goals are not empirical
selection thresholds or guarantees; document exceptions before experiments.

Reserve roughly 3–5 independent repository families for confirmation once the
overall cohort is feasible. Assign families before inspecting their evaluation
labels or predictions, using a recorded seed and a deterministic, documented
organization/domain balancing rule. Resolve the exact count/rule during Phase 2
design. Keep confirmation labels inaccessible to development tools, including
dataset-wide prevalence inventory; only label-free schema/coverage admission
checks are allowed for that reserved cohort.

The existing four-project audit is exploratory and its V6 outcomes are public.
Do not call any of those historical tests blind confirmation. Freeze all methods,
budgets, metrics and analysis rules before the confirmation evaluation. Once
opened, a confirmation cohort cannot remain a reusable tuning holdout.
