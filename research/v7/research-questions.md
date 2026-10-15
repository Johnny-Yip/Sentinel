# Research questions

Objective: investigate reliable defect-risk prediction across Java repositories
without labeled data from the held-out target entering development.

| Question | Proposed assessment | Hypothesis, not a result |
|---|---|---|
| RQ1: Can source-project-trained probability calibration improve reliability on unseen targets? | Paired raw/calibrated probability error and reliability by target/window; source-prior baseline | Source-only calibration may reduce probability error on some targets, but shift may defeat transfer |
| RQ2: Which policies are more stable under limited inspection budgets? | Precision/recall at declared per-project, per-date file budgets; budget violations and worst-project performance | Rank budgets may control workload more consistently than transferable probability cutoffs |
| RQ3: How robust are predictions across repositories, temporal windows and distributions? | Project/window effect distributions, worst cases, shift diagnostics and uncertainty | Performance variation may track domain and prevalence differences; association is not causation |

The outcome is **file-snapshot involvement in a heuristic bug-fix event in the
next 90 days**, not defect severity, introduced defects, or defects prevented.
The primary sampling unit for generalization is the repository; repeated rows
do not create additional independent repositories.

A lower Brier score alone does not establish better calibration; combine it with
reliability plots and ECE sensitivity. Source-only, prespecified choices are
essential. No minimum improvement or novelty claim is made. Unchanged dummy,
Logistic Regression and Random Forest baselines remain the algorithm family.

Budget initially means the count/fraction of files inspected at a decision date.
Current data has no measured review time or trustworthy snapshot LOC. Cumulative
churn must not be relabeled as inspection effort.
