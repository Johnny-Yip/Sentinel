# Structured literature review

Reviewed 2026-10-15. This is a bounded, reproducible seed review, not an exhaustive
systematic review or proof of novelty. Searches combined `cross-project defect
prediction benchmark`, `calibration distribution shift`, `effort-aware defect
prediction`, `time-based conclusion stability`, and `machine learning leakage`.
Sources were publisher/proceedings pages and author-hosted papers. Include
identifiable academic work relevant to one of the five themes; exclude unverifiable
citations and unsupported summaries. Read abstracts and relevant sections; no
claim is made that every paper was exhaustively replicated or reviewed.

## Evidence matrix

| ID / academic reference | Published finding supported by inspected material | Sentinel implication / applicability limit |
|---|---|---|
| L1. Steffen Herbold (2017), *A systematic mapping study on cross-project defect prediction*. [Author preprint](https://arxiv.org/abs/1705.06429) | Study setups differ in datasets, classifiers, metrics and baselines, limiting direct comparison | Keep matched populations and explicit baseline contracts; this does not establish a best Sentinel method |
| L2. Steffen Herbold, Alexander Trautsch, Jens Grabowski, *A Comparative Study to Benchmark Cross-Project Defect Prediction Approaches*, IEEE TSE 44(9), 811–833 (2018 issue; 2017 online DOI). [Author-hosted publication](https://www.swe.informatik.uni-goettingen.de/publications/comparative-study-benchmark-cross-project-defect-prediction-approaches), [DOI](https://doi.org/10.1109/TSE.2017.2724538) | Replicates 24 CPDP approaches across five datasets; highlights replication and evaluation heterogeneity | Broad benchmarks matter; its data and conclusions are not Sentinel's process-feature setting. Read the authors' [statistical correction](https://arxiv.org/abs/1707.09281) before reusing rankings |
| L3. Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger (2017), *On Calibration of Modern Neural Networks*, ICML/PMLR 70, 1321–1330. [Proceedings](https://proceedings.mlr.press/v70/guo17a.html) | Post-processing calibration, including temperature scaling, is effective in studied neural-network settings | Calibration is established; neural image/document findings do not guarantee benefits for class-weighted sklearn defect predictors |
| L4. Yaniv Ovadia et al. (2019), *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*, NeurIPS 32. [Proceedings](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) | Large-scale shifted-data experiments find traditional post-hoc calibration can fall short | Evaluate transfer rather than assume calibration invariance; the experiments are not CPDP evidence |
| L5. Sangdon Park, Osbert Bastani, James Weimer, Insup Lee (2020), *Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation*, AISTATS/PMLR 108, 3219–3229. [Proceedings](https://proceedings.mlr.press/v108/park20b.html) | Uses labeled source and unlabeled deployment data, importance weighting and a learned feature map to address covariate shift | Distinguish transductive adaptation from our source-only default. Neither weighting assumptions nor target adaptation is authorized or implemented here |
| L6. Steffen Herbold (2018), *Benchmarking cross-project defect prediction approaches with costs metrics*. [Author preprint](https://arxiv.org/abs/1801.04107) | Under its cost assumptions, cost-based method rankings differ from non-cost rankings and trivial baselines can be competitive | Include explicit budgets and simple baselines. This is an arXiv preprint, not claimed here as a peer-reviewed journal result; its cost model is not measured Sentinel effort |
| L7. Jonida Çarka, Matteo Esposito, Davide Falessi (2022), *On effort-aware metrics for defect prediction*, Empirical Software Engineering 27, 152. [Publisher / DOI](https://doi.org/10.1007/s10664-022-10186-7) | Studies metric usage and size-normalized ranking; shows that names/definitions and choice of effort metric matter | Define file budgets explicitly. Snapshot LOC and actual review time are absent, so size-aware or cost-saving claims are unsupported |
| L8. Abdul Ali Bangash, Hareem Sahar, Abram Hindle, Karim Ali (2020), *On the Time-Based Conclusion Stability of Cross-Project Defect Prediction Models*, Empirical Software Engineering 25, 5047–5083. [Author manuscript](https://softwareprocess.es/pubs/bangash2020EMSEstability.pdf), [DOI](https://doi.org/10.1007/s10664-020-09878-9) | Past-to-future evaluation reveals performance and conclusions varying across periods | Shared calendar roles and multiple prespecified windows matter; Sentinel must additionally enforce its own 90-day outcome maturity |
| L9. Sayash Kapoor, Arvind Narayanan (2023), *Leakage and the reproducibility crisis in machine-learning-based science*, Patterns, article 100804. [Publisher / DOI](https://doi.org/10.1016/j.patter.2023.100804) | Taxonomy of leakage and model info sheets supports checking claims and data dependencies | Record lineage, selection access and observation availability; this general ML evidence is not an audit of Sentinel |
| L10. Alexandru Niculescu-Mizil, Rich Caruana (2005), *Predicting Good Probabilities With Supervised Learning*, ICML. [Proceedings / DOI](https://doi.org/10.1145/1102351.1102430), [Conference paper](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf) | Compares probability quality across supervised learners and uses Platt scaling and isotonic regression to correct distortions; fitting on model-training predictions introduces bias | Sigmoid and isotonic calibration are established candidates, not inventions. Their finite-data results do not guarantee source-to-target transfer |

## Established methods versus proposed hypotheses

Established: cross-project benchmarks and data standardization/selection;
post-hoc sigmoid/isotonic calibration; effort-aware ranking evaluation; time-aware validation;
explicit leakage and reproducibility records. These are not V7 inventions.
Literature methods that alter representations, algorithms or use target data are
context, not automatically admissible V7 implementations.

Proposed Sentinel hypotheses: source-only calibration may improve probability
error for some targets; rank-budget policies may stabilize workload; project/time
shift may explain variation. None is established by the literature for our data.
Whether independent source calibration, policy selection and mature temporal
roles yield reliable transfer in this exact process-feature/label setting is an
empirical question. It is a potential study focus, not a demonstrated literature gap.

Open review work: systematic forward/backward citation searches, newer binary
calibration under label/concept shift, dependence-aware inference and explicit
effort-metric criticisms. Record search dates, exclusions and version corrections
before claiming coverage or novelty. No paper's reported improvement is imported
as a Sentinel result.
