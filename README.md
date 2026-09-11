# Sentinel

Sentinel is a machine-learning software engineering project intended to predict
which Java source files are most likely to be involved in future defects. V6
adds snapshot-level defect-risk scoring, individual prediction explanations,
actionable developer inspection guidance, and project-level risk intelligence
to V5's unified evaluation and experiment reports. It builds on
the V1 repository miner, V2 leakage-safe feature pipeline, V3 reproducible
sklearn baselines, and V4 cross-project evaluation. It does not expose an
application service or UI.

## Pipeline

```text
GitHub repository
        |
        v
raw Java commit history CSV
        |
        v
monthly file-level feature engineering
        |
        v
ML-ready snapshot CSV
        |
        v
temporally evaluated V3 baseline + artifacts
        |
        v
leave-one-project-out V4 report + artifacts
        |
        v
unified V5 evaluation report
        |
        v
ranked, explained, actionable, and project-level V6 risk intelligence
        |
        v
unified developer risk brief and inspection queue
```

The raw history contains one row per Java file changed in a commit. The V2
dataset contains one row per eligible Java file and monthly snapshot date, with
cumulative historical features and a future 90-day defect label.

## Install for development

Sentinel requires Python 3.11 or newer. Install the package and test dependencies
from the repository root:

```bash
python -m pip install -e '.[dev]'
```

## V1: mine Java history

Mine a public GitHub repository:

```bash
python -m sentinel.mine https://github.com/apache/commons-lang.git
```

This writes `commons-lang_java_history.csv` by default. Choose another path with
`--output`:

```bash
python -m sentinel.mine \
  https://github.com/apache/commons-lang.git \
  --output data/commons-lang_java_history.csv
```

Each raw row includes repository, commit hash, author, commit timestamp, commit
message, Java file path, lines added, and lines deleted.

## V2: build file snapshots

Transform a miner CSV into an ML-ready dataset:

```bash
python -m sentinel.features commons-lang_java_history.csv
```

For the example above, the default output is `commons-lang_features.csv`. An
explicit path can be provided with `--output`:

```bash
python -m sentinel.features \
  commons-lang_java_history.csv \
  --output data/commons-lang_features.csv
```

The command reports the number of snapshots, unique Java file paths, positive
and negative labels, the positive-label percentage, and the output location.

The output columns are:

- `repository`
- `file_path`
- `snapshot_date`
- `commit_count`
- `developer_count`
- `lines_added`
- `lines_deleted`
- `code_churn`
- `file_age_days`
- `days_since_last_change`
- `previous_bug_fixes`
- `defect_next_90_days`

`commit_count`, developer count, line totals, churn, age, recency, and previous
bug fixes are calculated only from changes on or before the snapshot date.
Developer identity uses normalized author email, falling back to normalized
author name when email is absent. `code_churn` is cumulative lines added plus
cumulative lines deleted.

### Exact snapshot strategy

- Commit timestamps are converted to UTC and snapshots are calendar month-ends.
- A path enters the dataset at the first month-end on or after its first observed
  Java change.
- It receives one snapshot at every later month-end for which the repository has
  a complete 90-day observation window.
- The latest Java change date in each repository's input CSV is treated as that
  repository's observation endpoint. Consequently, snapshots in the final 90
  days are omitted rather than assigning right-censored negative labels.
- Historical features include changes whose UTC calendar date is on or before
  the snapshot date.
- `defect_next_90_days` is `1` when the same repository/path appears in a
  bug-fix commit strictly after the snapshot date and no more than 90 days later;
  otherwise it is `0`.

The V1 schema does not record whether a change adds, deletes, or renames a path.
V2 therefore defines an active path as one that has been observed by the
snapshot date and carries it forward. Deleted and renamed paths cannot yet be
removed reliably; lifecycle-aware snapshots are a future mining improvement.

### Baseline bug-fix heuristic

Commit-message classification is deliberately isolated in
`sentinel.features.is_bug_fix_message` so an SZZ-based approach can replace it
later. The baseline matches conservative whole-word terms such as `fix`,
`fixed`, `fixes`, `bug`, `bugfix`, `defect`, `crash`, `incorrect`, `exception`,
`npe`, and `regression`, including ordinary plural forms. Whole-word matching
avoids broad substring matches such as `fixture`, `debug`, and `exceptional`.

The heuristic is a proxy: it can miss bug-fix commits with vague messages and
can identify fixes, but it does not determine which earlier change introduced a
defect.

## V3: train and evaluate ML baselines

V3 predicts whether a Java file snapshot will experience at least one bug-fix
event in the 90 days after its `snapshot_date`. It compares a prior-based dummy
baseline, class-weighted Logistic Regression, and a class-weighted Random
Forest. The configurations and random seeds are deterministic, and training is
separate from the V1 mining and V2 feature-engineering modules.

Install the project dependencies, then train all baselines with:

```bash
sentinel-train commons-lang_features.csv \
  --output-dir artifacts/commons-lang
```

The equivalent module command is:

```bash
python -m sentinel.ml commons-lang_features.csv \
  --output-dir artifacts/commons-lang \
  --random-state 42
```

When `--output-dir` is omitted, the default is
`artifacts/<dataset-name>`. The command validates the dataset, prints each
temporal split's date range and class distribution, trains the three models,
reports validation and test metrics at both 0.5 and a validation-selected
threshold, prints feature signals, and saves the validation-selected model.

### Why the split is temporal

A random row split would mix earlier and later snapshots of the same evolving
codebase. That would give training access to patterns from the future relative
to some evaluated rows and produce an unrealistically optimistic estimate.
V3 instead sorts the distinct snapshot dates and assigns the earliest 70% to
training, the next 15% to validation, and the final 15% to test. Every row with
the same `snapshot_date` stays in one partition.

Preprocessing and model fitting use only the training period. The validation
period selects each probabilistic decision threshold by maximum F1 and selects
the best model primarily by PR-AUC. The test period is evaluated only after
those choices are locked.

### Why PR-AUC matters

Only about 1.97% of the current commons-lang snapshots are positive. A model
that predicts every snapshot as negative would therefore be about 98% accurate
while finding no future bug-fix events. Accuracy is not reported as the primary
metric. Sentinel emphasizes PR-AUC (Average Precision), which summarizes the
precision/recall tradeoff for the rare positive class, while also reporting:

- accuracy;
- precision, recall, and F1;
- ROC-AUC;
- the confusion matrix; and
- the positive prediction rate.

Compare a model's PR-AUC with the positive prevalence in the evaluated period:
a no-skill ranking is near that prevalence. Recall answers how many positive
snapshots were found; precision answers how many alerts were actually positive;
and F1 balances the two at a particular threshold. The selected threshold is a
validation decision rule, not a test-tuned value.

Logistic Regression coefficients are based on standardized numeric features.
Positive coefficients increase the model's defect-risk score and negative
coefficients decrease it, holding other features fixed. Random Forest
importances show how much each feature contributed to tree splits, not the
direction of effect. Neither should be interpreted as causal evidence.

### Features and leakage controls

V3 uses an explicit whitelist of the eight historical V2 predictors:
`commit_count`, `developer_count`, `lines_added`, `lines_deleted`,
`code_churn`, `file_age_days`, `days_since_last_change`, and
`previous_bug_fixes`. It never trains on `repository`, `file_path`,
`snapshot_date`, `defect_next_90_days`, extra identifiers, or unrecognized
columns. The whitelist also prevents a newly added future-derived field from
silently entering a V3 model.

The V3 leakage audit passes the following controls:

- no random splitting is used, and rows sharing a snapshot date remain grouped;
- all preprocessors and models are fitted on training rows only;
- validation labels alone select thresholds and the winning model;
- test labels never influence preprocessing, fitting, threshold selection, or
  model selection, and test metrics are computed only after selection is locked;
- only the historical V2 feature whitelist is passed to sklearn; and
- the future label, repository/path/date metadata, identifiers, unrecognized
  columns, and future-derived fields are excluded from predictors.

The same audit result and control settings are recorded in each generated
`metadata.json`.

The saved artifact directory contains:

```text
artifacts/commons-lang/
├── model.joblib
└── metadata.json
```

`model.joblib` is the complete fitted sklearn pipeline. `metadata.json` records
the Sentinel version, model type, exact feature list, excluded columns, random
state, train/validation/test ranges, validation-selected threshold, evaluation
metrics, feature signals, and all-model comparison. Only load joblib files from
trusted sources.

## V4: cross-project generalization

Cross-project generalization asks whether a model trained on some repositories
can rank defect risk in a repository it has never seen. This is stricter than
V3's same-project temporal evaluation: V3 learns from earlier snapshots of the
same codebase, while V4 excludes every row from the target repository until the
single final held-out evaluation.

V4 uses leave-one-project-out evaluation. For each repository in turn, that
entire repository is held out. Within every remaining repository, the earliest
80% of distinct snapshot dates become training data and the latest 20% become
validation data. Rows sharing a date within a repository always remain
together. The training portions and validation portions are then concatenated
separately. Preprocessing and model fitting use only training rows, while
validation PR-AUC selects the model and validation F1 selects each model's
threshold. The held-out repository is evaluated once after those choices are
locked.

### Generate multiple repository datasets

Mine each public Java repository with V1 and transform it with V2. For example:

```bash
sentinel-mine https://github.com/apache/commons-lang.git \
  --output data/commons-lang_java_history.csv
sentinel-features data/commons-lang_java_history.csv \
  --output data/commons-lang_features.csv

sentinel-mine https://github.com/apache/commons-io.git \
  --output data/commons-io_java_history.csv
sentinel-features data/commons-io_java_history.csv \
  --output data/commons-io_features.csv

sentinel-mine https://github.com/apache/commons-collections.git \
  --output data/commons-collections_java_history.csv
sentinel-features data/commons-collections_java_history.csv \
  --output data/commons-collections_features.csv

sentinel-mine https://github.com/junit-team/junit4.git \
  --output data/junit4_java_history.csv
sentinel-features data/junit4_java_history.csv \
  --output data/junit4_features.csv
```

V2 writes the explicit `repository` metadata field on every row. The tuple
`repository`, `file_path`, and `snapshot_date` must uniquely identify each
snapshot. V4 validates every input and rejects missing predictors, malformed
values, duplicate snapshots, repeated repositories across input files, fewer
than two repositories, and repositories that cannot support chronological
training/validation splitting.

### Run leave-one-project-out evaluation

```bash
sentinel-cross-project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  data/junit4_features.csv \
  --output-dir artifacts/cross-project \
  --random-state 42
```

The equivalent module command is:

```bash
python -m sentinel.cross_project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  data/junit4_features.csv \
  --output-dir artifacts/cross-project \
  --same-project-metadata artifacts/commons-lang/metadata.json
```

The same three centralized V3 model configurations are reused:
`DummyClassifier`, Logistic Regression, and `RandomForestClassifier`. No
repository indicator is ever passed to sklearn. The only predictors are the
eight documented historical V2 features. Repository, file path, snapshot date,
target label, future-derived fields, identifier-like fields, and unknown
metadata columns are excluded by an explicit whitelist. This matters because a
repository identifier would let a model memorize project-level prevalence
rather than learn transferable software-engineering signals.

### Metrics and interpretation

Each model is reported at the default threshold `0.5` and its locked
validation-selected threshold. Held-out metrics include accuracy, precision,
recall, F1, ROC-AUC, PR-AUC/Average Precision, confusion matrix, positive
prediction rate, class prevalence, and PR-AUC lift:

```text
PR-AUC lift = held-out PR-AUC / held-out positive rate
```

Lift makes projects with different defect prevalence easier to compare. A lift
above `1.0` means ranking performance is above the held-out repository's
prevalence baseline; it does not by itself imply that precision or recall is
operationally sufficient. Fold-to-fold variation should be treated as evidence
of repository sensitivity. A high mean with a large standard deviation or a
poor minimum is less robust than consistently positive lift.

The CLI output follows this format, with values filled by the actual run:

```text
Held out: <repository>
Train/validation repositories: <all other repositories>
<model>: validation PR-AUC=<value>, selected threshold=<value>
  test @ 0.5: precision=..., recall=..., F1=..., ROC-AUC=..., PR-AUC=..., lift=...
  test @ selected: precision=..., recall=..., F1=..., ROC-AUC=..., PR-AUC=..., lift=...
```

The aggregate table contains each model's mean, median, standard deviation,
minimum, and maximum held-out PR-AUC, plus mean lift, F1, precision, recall, and
ROC-AUC. Logistic Regression standardized coefficients are aggregated by mean,
standard deviation, and sign consistency. Random Forest importances are
aggregated across folds. Dataset shift compares training-repository and
held-out medians/IQRs for every predictor and includes an optional robust median
shift in training-IQR units.

When `artifacts/commons-lang/metadata.json` is present, V4 also compares its
commons-lang held-out Logistic Regression metrics with the recorded V3
same-project test metrics. The comparison reports PR-AUC, F1, and ROC-AUC
differences without using V3 data for V4 fitting or selection.

The V4 artifact directory is separate from V3 and contains:

```text
artifacts/cross-project/
├── folds.csv
├── aggregate.csv
├── model_selection.json
├── selected_thresholds.csv
├── coefficient_stability.csv
├── feature_importance_stability.csv
├── dataset_shift.csv
├── report.json
└── report.md
```

`folds.csv` contains every model at both thresholds. `report.json` is the full
machine-readable experiment record, including repository statistics, fold
definitions, validation choices, generalization interpretation, same-project
comparison, and the per-fold leakage audit. `report.md` is a concise human
summary. V4 does not overwrite V3's `model.joblib` or `metadata.json`.

The dedicated V4 leakage audit must pass for artifacts to be produced. It checks
repository disjointness, per-repository date grouping, the historical feature
whitelist, metadata/label/future-field exclusion, train-only preprocessing,
validation-only threshold and model selection, and final-evaluation-only use of
held-out labels.

## V5: unified evaluation and reporting

V5 gives within-project and cross-project experiments one report contract. It
reuses the existing V3 model registry, chronological splitting, V4
leave-one-project-out folds, validation-only threshold selection, and leakage
controls. It does not introduce a second training implementation.

Run a within-project evaluation with:

```bash
sentinel-evaluate within-project data/commons-lang_features.csv \
  --output-dir reports/commons-lang-within \
  --random-state 42
```

Run a cross-project evaluation with:

```bash
sentinel-evaluate cross-project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  --output-dir reports/cross-project \
  --random-state 42
```

The equivalent module entry point is `python -m sentinel.evaluation`. When
`--output-dir` is omitted, within-project reports use
`reports/<dataset-name>-within-project` and cross-project reports use
`reports/cross-project`.

Each experiment directory contains the common V5 report artifacts:

```text
reports/<experiment>/
├── metrics.json
├── model_comparison.csv
├── feature_importance.csv
├── confusion_matrix.png
└── report.md
```

`metrics.json` contains the dataset summary, model metrics, selected thresholds,
and evaluation strategy. `model_comparison.csv` uses the same accuracy,
precision, recall, F1, ROC-AUC, and PR-AUC pipeline for every model. For
cross-project evaluation it reports aggregates over held-out folds.

`feature_importance.csv` has one normalized schema for every model. Linear
models use native coefficients, tree models use native feature importance, and
models without either interface use deterministic permutation importance. The
file records the method so values with different semantics are not mistaken for
directly comparable effects. Cross-project values are aggregated across folds.
Explainability is calculated only after model and threshold selection are
locked; it is descriptive and does not alter selection.

For within-project evaluation, `confusion_matrix.png` shows the
validation-selected model on the temporal test split. For cross-project
evaluation, it pools the final confusion matrices from the model selected in
each held-out fold. `report.md` summarizes the dataset, comparison, top feature
signals, and key findings.

## V6 Phase 1: defect-risk scoring and ranking

V6 adds a `risk_score` for every snapshot that is part of final evaluation:
the temporal test split in within-project mode and each repository's held-out
fold in cross-project mode. The score is the selected model's positive-class
probability when `predict_proba` is available. For compatible estimators that
only expose `decision_function`, Sentinel deterministically maps the signed
margin through a sigmoid. All scores are therefore bounded in `[0, 1]`; the
fallback is useful for ordering but is not a calibrated probability.

Samples are sorted from highest to lowest risk. Equal scores use project, file
path, snapshot date, model name, and original row order as deterministic
tie-breakers. `predicted_label` uses the model's validation-selected decision
threshold. The separate risk threshold controls only the high-risk count and
top list in the summary; it does not change predictions or model metrics.

Both existing `sentinel-evaluate` modes create two additional files alongside
all five V5 artifacts:

- `risk_ranking.csv` contains rank, risk score, predicted and true labels,
  project, file path, snapshot date, selected model, scoring method, evaluation
  mode, and decision threshold.
- `risk_summary.json` contains the evaluated and high-risk sample counts, mean
  and maximum scores, the configured cutoff, top high-risk samples, and
  model/mode metadata.

The defaults are the top 10 rows at a high-risk cutoff of `0.5`. They can be
changed for either mode:

```bash
sentinel-evaluate within-project data/commons-lang_features.csv \
  --output-dir reports/commons-lang-within \
  --top-risk 25 \
  --risk-threshold 0.7

sentinel-evaluate cross-project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  --output-dir reports/cross-project \
  --top-risk 50 \
  --risk-threshold 0.65
```

In cross-project reports, independently selected fold models may have different
decision thresholds. The CSV records the model and threshold used for each row
so that the pooled ranking remains auditable.

## V6 Phase 2: individual prediction explanations

Phase 2 explains every sample in the final evaluation population: the temporal
test split for within-project evaluation and every held-out repository row for
cross-project evaluation. This is local explanation, which answers why the
selected fitted model assigned one snapshot its risk score. It differs from
`feature_importance.csv`, which summarizes a feature's overall model-level
importance and cannot explain a particular prediction.

For Logistic Regression, Sentinel multiplies each fitted coefficient by the
sample's transformed value after the training-fitted scaler. These exact
additive contributions are in log-odds space. For classifier trees and Random
Forests, Sentinel attributes the positive-class probability change along the
sample's decision path to each split feature and averages across trees. Other
estimators use a deterministic local fallback: replace one raw feature at a
time with that evaluation partition's median and measure the risk-score change.
No SHAP dependency is required.

A positive contribution (`increases_risk`) pushes the prediction toward higher
defect risk relative to that method's baseline; a negative contribution
(`decreases_risk`) is protective; an exact zero is `neutral`. Raw contribution
units differ by method and should not be compared across model families.
`normalized_contribution` is the feature's share of the sample's total absolute
contribution magnitude, so non-zero shares sum to approximately `1` for each
sample. `feature_value` is the raw value when preprocessing is one-to-one, and
`model_input_value` records the actual transformed value used by the model.

Both `sentinel-evaluate` modes now also generate:

- `prediction_explanations.csv`, with one row per explained feature and sample,
  including identifiers, prediction fields, raw and normalized contributions,
  direction, method, and deterministic local rank;
- `explanation_summary.json`, with explained row counts, common increasing and
  decreasing features, average absolute contribution by feature, and model and
  method metadata; and
- four concise columns in `risk_ranking.csv`: the strongest risk-increasing and
  risk-decreasing feature and contribution for each sample.

All eight historical predictors are explained by default. Limit only the
detailed CSV rows (while retaining the strongest positive and negative ranking
drivers) with `--explain-top-k`:

```bash
sentinel-evaluate within-project data/commons-lang_features.csv \
  --output-dir reports/commons-lang-within \
  --explain-top-k 5

sentinel-evaluate cross-project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  --output-dir reports/cross-project \
  --explain-top-k 5
```

## V6 Phase 3: Actionable Risk Insights

Phase 3 translates Phase 2's local contributions into concise, developer-facing
inspection guidance. It does not calculate model explanations again. Each of the
eight supported historical features has a stable category, human-readable
interpretation, and practical recommendation. Positive contributions are risk
signals and negative contributions are protective signals; this describes the
fitted model's local behavior, not a claim that a larger raw feature value is
always harmful or helpful.

For every evaluated snapshot, Sentinel selects up to three strongest non-neutral
supported contributions using absolute magnitude and feature name as a stable
tie-breaker. Insight generation always uses the full explanation set, so
`--explain-top-k` limits only `prediction_explanations.csv` and cannot change the
ranking guidance. Unknown future features are skipped rather than assigned
guessed advice and are listed in artifact metadata.

`risk_ranking.csv` now includes three concise fields:

- `primary_risk_reason`: interpretation of the strongest risk-increasing signal,
  or an explicit fallback when no such signal exists;
- `recommended_action`: the corresponding developer inspection prompt; and
- `risk_signal_count`: number of supported positive contributions for the
  sample.

Both within-project and cross-project evaluations also write
`actionable_insights.json`. It contains the per-sample selected risk and
protective signals, aggregate counts, common signal categories, feature
definitions, and deterministic-method metadata. A shortened sample looks like:

```json
{
  "schema_version": 1,
  "total_samples": 24,
  "common_risk_signals": [
    {
      "feature": "code_churn",
      "category": "high_churn",
      "sample_count": 11
    }
  ],
  "samples": [
    {
      "sample_id": "owner/project|src/Parser.java|2024-01-31",
      "primary_risk_reason": "Overall code churn is pushing predicted risk upward.",
      "recommended_action": "Inspect churn hotspots for repeated rewrites, broad diffs, and weak regression coverage.",
      "risk_signal_count": 3,
      "protective_signal_count": 2,
      "insights": [
        {
          "feature": "code_churn",
          "category": "high_churn",
          "signal_type": "risk"
        }
      ]
    }
  ]
}
```

## V6 Phase 4: Project-Level Risk Intelligence

Phase 4 aggregates the final-evaluation predictions and the already-computed
Phase 2/3 explanations and insights into deterministic engineering summaries.
It does not fit another model, recompute attribution, or create a second
definition of high risk. Every high-risk count uses the same
`--risk-threshold` cutoff as `risk_summary.json`; this summary cutoff remains
separate from the validation-selected threshold that produces model labels and
metrics.

Each V6 evaluation directory now also contains:

- `project_intelligence.json`, a stable schema with project summary statistics,
  risk concentration, recurring file hotspots, risk/protective signal profiles,
  descriptive temporal trends, top priorities, and method metadata; and
- `developer_priority.csv`, the complete deterministic sample inspection queue.

The JSON's principal fields are:

- `summary`: evaluated/high-risk counts, high-risk percentage, mean, median and
  maximum predicted risk, five fixed risk-score bands, dominant signals, and
  the number of identifiable risky files;
- `risk_concentration`: risk share from the top 10% and 20% of samples, the
  sample count needed to reach 50% of aggregate predicted risk, and
  concentration in repeatedly high-risk file paths;
- `hotspots`: file paths with at least two observations at or above the risk
  cutoff, ordered by repeated high-risk count and stable tie-breakers;
- `risk_signal_profile`: common and strongest aggregate risk/protective
  contributions, using the existing Phase 3 feature registry and recording
  contribution methods/spaces;
- `temporal_analysis`: calendar-month mean risk, high-risk rate, dominant risk
  signal, and change from the prior available period; and
- `metadata`: formulas, tie-breaks, threshold semantics, unsupported analyses,
  unknown features, identifier/timestamp availability, and limitations.

### Concentration definitions

Sentinel sorts risk scores descending. The top-10% share is the sum of the
highest `ceil(0.10 * sample_count)` scores divided by the sum of all scores; the
top-20% share uses `ceil(0.20 * sample_count)`. The 50% count is the smallest
descending-score prefix whose sum reaches at least half of aggregate predicted
risk. If aggregate predicted risk is zero, both shares and the 50% count are
defined as zero. These ceiling rules make one-row and other small datasets
well-defined.

Repeated-entity concentration uses every observation of any available
`file_path` with at least two high-risk observations. Sentinel reports those
observations' share of aggregate predicted risk and their share of identifiable
high-risk observations. If paths are unavailable, hotspot analysis is marked
unsupported rather than inventing identifiers.

### Developer priority score

The queue uses this transparent inspection heuristic:

```text
recurring_hotspot_evidence = high_risk_rate * min(high_risk_count / 2, 1)
priority_score = 0.80 * predicted_risk + 0.20 * recurring_hotspot_evidence
```

`high_risk_count` and `high_risk_rate` are calculated over observations of the
same available file path in the project. The recurrence multiplier reaches its
maximum at two high-risk observations. Rows are ordered by priority score,
predicted risk, recurring high-risk count, identifier, snapshot date, and
source order. In cross-project mode, ranks restart at 1 inside each held-out
project. A priority score is an inspection heuristic, not a probability.

Run a within-project report exactly as in earlier V6 phases:

```bash
sentinel-evaluate within-project commons-lang_features.csv \
  --output-dir reports/commons-lang-within \
  --risk-threshold 0.7
```

The resulting `project_intelligence.json` has a `single_project` summary and
`report.md` includes a concise `Project Risk Intelligence` section. For
cross-project/LOPO evaluation:

```bash
sentinel-evaluate cross-project \
  commons-lang_features.csv \
  commons-io_features.csv \
  --output-dir reports/commons-cross-project \
  --risk-threshold 0.7
```

The cross-project artifact uses `summary.analysis_scope = "per_project"` and a
`projects` array. Each held-out project has its own summary, concentration,
hotspots, signal profile, temporal analysis, and priority ranks. Sentinel does
not pool unrelated held-out projects into a misleading project-level mean or
concentration statistic.

Project intelligence summarizes fitted-model signals. Risk signals are
associations, not causal claims; temporal movements are descriptive and do not
claim statistical significance. Contribution totals should only be compared
within the same explanation method and contribution space. Sentinel does not
prove that a file contains a defect, and missing timestamps or identifiers are
reported explicitly instead of being fabricated.

## V6 Phase 5: Unified Developer Risk Brief

Phase 5 turns the existing Phase 1–4 outputs into one concise decision layer for
developers. It consumes `risk_ranking.csv`, the full in-memory prediction
explanations, `actionable_insights.json`, `project_intelligence.json`,
`developer_priority.csv`, the feature insight registry, and available model
metadata. It does not fit a model, rescore a sample, choose a threshold, or
recompute an explanation.

Every unified evaluation now adds:

- `developer_risk_brief.json`, the stable machine-readable project brief;
- `developer_risk_brief.md`, a standalone developer-facing rendering; and
- `developer_actions.csv`, the complete queue of samples at or above the
  existing configured risk cutoff.

The earlier V5 and V6 artifacts remain unchanged. `report.md` also ends with a
concise `Developer Risk Brief` section containing the attention level,
concentration result, recurring hotspot, temporal direction, highest-risk
signal, up to five actions per project, and an interpretation warning.

### Risk tier definitions

Tiers are project-local inspection priorities. Let `H` be the number of samples
in one project whose `risk_score >= risk_threshold`. Sort those samples by the
existing Phase 4 priority score, predicted risk, recurring-hotspot evidence,
identifier, timestamp, existing reason/action text, and signal count. Then:

- `CRITICAL`: the first `max(1, ceil(0.10 * H))` priority items;
- `HIGH`: the remaining items through `ceil(0.30 * H)`;
- `MEDIUM`: all remaining items at or above the cutoff; and
- `LOW`: samples below the cutoff.

When `H` is zero, every evaluated sample is `LOW` and there are no developer
actions. Sparse projects need not contain every tier. Ranks and tier boundaries
restart within each project. The names express relative inspection urgency;
they are not calibrated defect probabilities or defect-severity labels.

### Attention level definitions

The executive-summary attention level is a deterministic workload heuristic
based on the fraction of project samples at or above the same configured risk
cutoff:

- `LOW`: no samples meet the cutoff;
- `MODERATE`: a non-zero share below 10% meets it;
- `ELEVATED`: at least 10% but less than 25% meets it; and
- `HIGH`: at least 25% meets it.

The cutoff remains the Phase 1 `--risk-threshold` summary setting. Neither the
cutoff nor the attention level replaces the validation-selected classification
threshold, and neither is a calibrated probability.

### Artifact schemas

`developer_risk_brief.json` contains:

```text
schema_version
experiment_type
analysis_scope
project_count
projects[]
  project
  executive_summary
    total_samples
    high_risk_count
    high_risk_rate
    mean_risk
    concentration_summary
    top_recurring_hotspot
    temporal_direction
    highest_risk_signal
    number_of_priority_items
    overall_attention_level
    risk_tier_counts
  top_developer_actions[]
  models_used
metadata
  input_dependencies
  risk_tier_definitions
  attention_level_definitions
  tie_breaking_rules
  feature_registry
  model_metadata
  limitations
```

Each entry in `top_developer_actions` contains `project`, `priority_rank`,
`risk_tier`, `identifier`, `file_path`, `snapshot_date`, `predicted_risk`,
`priority_score`, `short_reason`, `dominant_signal`, `recommended_action`, and
`supporting_evidence`. The top list is limited to five per project.
`developer_actions.csv` uses the same columns for the complete high-risk action
queue and stores `supporting_evidence` as deterministic JSON text. Missing file
paths, timestamps, signals, and explanation evidence remain empty/null; Sentinel
does not manufacture replacements.

Recommendations are copied from Phase 3's stable feature registry and are
phrased as inspection, review, ownership, or testing suggestions. They are not
claims that a defect exists. Supporting evidence records only available model
outputs, such as the risk score and cutoff, recurring high-risk observations,
dominant feature contribution, explanation method, and contribution space.

### Within-project example

```bash
sentinel-evaluate within-project data/commons-lang_features.csv \
  --output-dir reports/commons-lang-within \
  --risk-threshold 0.7 \
  --random-state 42
```

This creates one project brief. Its action ranks and tiers apply only to the
temporal test population for that project.

### Cross-project example

```bash
sentinel-evaluate cross-project \
  data/commons-lang_features.csv \
  data/commons-io_features.csv \
  data/commons-collections_features.csv \
  --output-dir reports/cross-project \
  --risk-threshold 0.7 \
  --random-state 42
```

This creates a separate brief for every held-out project. Each section has its
own tier counts, attention level, hotspot, temporal description, and action
ranks. Sentinel does not pool project risk levels or compare absolute rankings
across independently selected held-out models.

Phase 5 inherits the limitations of its inputs: scores may be uncalibrated;
feature contributions are model descriptions rather than causal explanations;
the V1 schema cannot reliably retire deleted or renamed paths; small projects
produce coarse relative tiers; and missing identifiers or timestamps limit
hotspot and temporal analysis. The brief supports prioritization, not automated
defect adjudication.

## Tests

Run the full V1 through V6 test suite:

```bash
pytest
```
