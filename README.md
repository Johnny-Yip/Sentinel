# Sentinel

Sentinel is a machine-learning software engineering project intended to predict
which Java source files are most likely to be involved in future defects. V5
adds unified evaluation, explainability, and experiment reports to the V1
repository miner, V2 leakage-safe feature pipeline, V3 reproducible sklearn
baselines, and V4 cross-project evaluation. It does not expose an application
service or UI.

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

## Tests

Run the full V1 through V5 test suite:

```bash
pytest
```
