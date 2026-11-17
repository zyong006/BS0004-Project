# Appendix: more rigorous machine learning on the Golub ALL vs AML data

All numbers below come from `results/` (seed 2026) and `results/seed2028/`. They were produced by one run of `appendix_advanced_ml.R`.

- **Positive class:** AML.
- **Cross-validation (CV):** "CV" means nested CV on the 38 training patients: an outer loop of stratified 5-fold CV repeated 20 times, and an inner loop of stratified 5-fold CV inside each outer training fold.
- **CV intervals:** given as the mean, with the 2.5th to 97.5th percentile across the 20 repeats in brackets.
- **Test-set intervals:** stratified bootstrap 95% confidence intervals (CI) from 2000 resamples.

## 0. Starting point and checks

- **Original script:** the original script was read and not modified. It was not on this computer at the path given, so the copy pasted into the conversation was used.
- **Preprocessing matches the original:** the appendix reimplements the same preprocessing. The script stops if the top-10 fold-change probes it finds differ from the original top-10. They matched exactly, and 3051 genes passed the variation filter, as in the report.
- **Original leaky estimate reproduced** (`A1_original_caret_leaky_cv.csv`): genes were ranked on all 38 patients, then caret's 5-fold CV was run. The CV AUC was 1.000 for every odd k from 1 to 15, so caret's tie-break picked k = 15. This is identical to the report.
- **Gene symbols:** the Bioconductor package `hu6800.db` was not installed (the install was stopped). Symbols were therefore taken from the NCBI GEO annotation of the Hu6800 array (GPL80), saved in `annotation/GPL80.annot.gz`.
  - In that annotation, probe D88270_at maps to **VPREB1**; the report's Table 2 calls it IGL.

## 1. Nested cross-validation: leaky vs honest KNN, elastic net, shrunken centroids

`A2_nested_cv_performance.csv` (seed 2026):

| Model | Cut-off | AUC | Sensitivity (AML recall) | Specificity | Balanced accuracy |
|---|---|---|---|---|---|
| KNN, leaky (genes ranked on all 38) | 0.5 | 0.977 (0.955-1.000) | 0.936 (0.818-1.000) | 1.000 (1.000-1.000) | 0.968 (0.909-1.000) |
| KNN, leaky | Youden | 0.977 | 0.950 (0.909-1.000) | 1.000 | 0.975 (0.955-1.000) |
| KNN, honest (genes ranked in fold) | 0.5 | 0.969 (0.931-1.000) | 0.782 (0.589-0.957) | 0.994 (0.963-1.000) | 0.888 (0.786-0.978) |
| KNN, honest | Youden | 0.969 | 0.932 (0.814-1.000) | 0.972 (0.926-1.000) | 0.952 (0.879-1.000) |
| Elastic net (primary) | 0.5 | 0.997 (0.991-1.000) | 0.955 (0.861-1.000) | 0.981 (0.944-1.000) | 0.968 (0.913-1.000) |
| Elastic net | Youden | 0.997 | 0.977 (0.909-1.000) | 0.974 (0.944-1.000) | 0.976 (0.945-1.000) |
| Shrunken centroids | 0.5 | 0.981 (0.951-1.000) | 0.923 (0.818-1.000) | 0.933 (0.850-1.000) | 0.928 (0.867-0.991) |
| Shrunken centroids | Youden | 0.981 | 0.914 (0.770-1.000) | 0.941 (0.850-1.000) | 0.927 (0.853-0.973) |

What this shows:

- **The original CV AUC of 1.000 was optimistic, but the honest AUC is still high.** Ranking genes inside each fold gives a CV AUC of 0.969 for KNN. The honest and leaky AUC intervals overlap (0.931-1.000 vs 0.955-1.000), so the AUC gap alone is not convincing.
- **The leak shows up mainly in sensitivity and in how k was chosen:**
  - At the 0.5 cut-off, honest KNN caught 0.782 of AML in CV, against 0.936 for the leaky version.
  - In the leaky version, every one of the 100 outer fits had tied inner AUCs across k values. In the honest version, 69 of 100 did (`A3_nested_cv_tuning.csv`).
  - The leaky version chose k = 1 or 3 in 90 of 100 fits. The honest version spread its choices from k = 1 to 13.
- **Averaging over folds is not the explanation.** caret averages the AUC over folds of about 8 patients, whereas the table above pools all out-of-fold predictions. The two give almost the same answer (`A4_pooled_vs_fold_averaged_auc.csv`): 0.977 vs 0.977 for leaky KNN, and 0.969 vs 0.973 for honest KNN.
- **Remaining differences from the original 1.000:**
  - caret reports a separate AUC for each fixed k. Here, k itself is tuned inside every fold.
  - The original z-scored all 38 patients before CV.
  - The original used one split, not 20 repeats.
  - This appendix does not separate the contributions of these three differences.
- **Elastic net, the pre-declared primary model, had the highest and most stable CV AUC:** 0.997, with a standard deviation (SD) of 0.003 across repeats.
  - Shrunken centroids: 0.981 (SD 0.017).
  - Honest KNN: 0.969 (SD 0.025).
  - Shrunken centroids and honest KNN overlap with each other and with the top of the elastic net range.
- **Constraint on k:** k was limited to odd values with k < 2 x (AML patients in the smallest inner fitting set), so that AML can win a majority vote. This kept k at 13 or below inside the outer folds.

## 2. Decision threshold

The Youden threshold was always chosen from inner-fold out-of-fold probabilities of the training data. Ties were broken by taking the cut-off closest to 0.5.

- **KNN:** the threshold was the main fix, as expected from problem 3 (imbalance).
  - Honest KNN's CV sensitivity rose from 0.782 to 0.932.
  - Specificity fell only slightly, from 0.994 to 0.972.
  - The median tuned cut-off was 0.207, far below 0.5.
- **Elastic net:** it was already trained with class weights, so tuning the threshold changed little. CV sensitivity went from 0.955 to 0.977 with seed 2026, and from 0.973 to 0.964 with seed 2028.
- **Shrunken centroids:** these use equal class priors, and tuning the threshold made no consistent difference.

## 3. Independent test set (34 patients, each model used once)

`D2_test_performance_bootstrap_ci.csv` and `D3_test_confusion_matrices.csv` (seed 2026):

| Model | Cut-off | AUC (95% CI) | AML recall (95% CI) | Specificity | Accuracy (95% CI) | TP/FN/TN/FP |
|---|---|---|---|---|---|---|
| KNN, honest (k = 7) | 0.5 | 0.998 (0.989-1.000) | 0.929 (0.786-1.000) | 1.000 | 0.971 (0.912-1.000) | 13/1/20/0 |
| KNN, honest | 0.214 | same | 0.929 (0.786-1.000) | 1.000 | 0.971 | 13/1/20/0 |
| Elastic net (83 genes) | 0.5 | 0.996 (0.979-1.000) | 0.857 (0.643-1.000) | 1.000 | 0.941 (0.853-1.000) | 12/2/20/0 |
| Elastic net | 0.380 | same | 0.929 (0.786-1.000) | 1.000 | 0.971 (0.912-1.000) | 13/1/20/0 |
| Shrunken centroids (3051 genes) | 0.5 | 0.993 (0.968-1.000) | 0.786 (0.571-1.000) | 1.000 | 0.912 (0.824-1.000) | 11/3/20/0 |
| Shrunken centroids | 0.489 | same | 0.786 (0.571-1.000) | 1.000 | 0.912 | 11/3/20/0 |

For comparison, the original KNN (50 genes, k = 15, cut-off 0.5) caught 9 of 14 AML (TP 9, FN 5, TN 20, FP 0), with test AUC 0.991.

- **No winner is declared, and the data would not support one.** All three test AUCs lie between 0.993 and 0.998, and their CIs overlap almost completely. Recall CIs also overlap (lower limits 0.571-0.786, all reaching 1.000).
- **All three honest models caught more AML at 0.5 than the original 9/14.** They caught 11-13 of 14, with no false positives at seed 2026.
- **The leaky KNN row is not a different model on the test set.** Its final model (`D1_final_models.csv`) is also trained on all 38 patients with the same 50 genes, so it differs from honest KNN only in the k and cut-off it chose. It caught 13/14 at 0.5 and 14/14 at its Youden cut-off.
- **Shrunken centroids at seed 2026 used all 3051 genes.** The inner CV error was 1/38 with no shrinkage and 2/38 at every shrinkage value up to 1.29. The pre-declared rule (lowest error) therefore picked no shrinkage on a one-patient difference. With seed 2028 the same rule picked shrinkage 1.075 (886 genes). The amount of shrinkage is not stable on 38 patients.

**Who was misclassified** (`D5_misclassified_test_patients.csv`):

- **All errors at seed 2026 came from hospitals with no training samples.** The training samples came from DFCI and CALGB only. At seed 2026, every misclassified test patient, for every model and both cut-offs, came from St-Jude or CCG.
- **Patient 66 was missed by every honest model at the 0.5 cut-off, with both seeds.** This patient is AML, bone marrow, from CCG; elastic net scored them 0.307. They were caught only at very low cut-offs: by leaky KNN at its Youden cut-off of 0.167 (seed 2026), and by elastic net at its Youden cut-off of 0.257 (seed 2028).
- **At seed 2028, one training-hospital patient was also wrong.** Patient 67 (ALL, blood, DFCI) became a false positive for honest KNN and elastic net at their tuned cut-offs, and for shrunken centroids at both cut-offs.
- **In the training set, hospital and diagnosis are completely confounded.** All 27 ALL come from DFCI and all 11 AML from CALGB, so a model cannot separate "AML" from "processed at CALGB". In the test set, AML from the two new hospitals got lower elastic net probabilities than AML from CALGB: median 0.84 vs 0.99, Wilcoxon p = 0.034. St-Jude contributes both classes, and its 3 ALL were still scored as clearly ALL (0.001-0.014), so the model is not detecting hospital alone. The confounding does, however, plausibly explain why the misses are all AML from new hospitals.

## 4. Permutation test (elastic net, full nested CV rerun on each permutation)

`E2_permutation_summary.csv`:

| Seed | Permutations | Observed mean CV AUC | Null mean (SD) | Null 95th percentile | Null maximum | Empirical p |
|---|---|---|---|---|---|---|
| 2026 | 200 | 0.997 | 0.489 (0.063) | 0.632 | 0.727 | 0.005 |
| 2028 | 200 | 0.997 | 0.490 (0.067) | 0.639 | 0.793 | 0.005 |

- **The null is centred on chance.** That is also a check against leakage: a pipeline that leaked labels would give permuted-label AUCs above 0.5.
- **No permutation came close to the observed AUC.** p = 1/201 is the smallest value possible with 200 permutations.

## 5. Gene selection stability (100 stratified 80% subsamples)

`F_gene_selection_frequency.csv` and `F_original_top10_selection.csv` (seed 2026):

- **Elastic net** selected a median of 81 probes per fit; 25 probes were selected in at least 90% of subsamples.
  - Probes selected in 100% of subsamples: CST3, ZYX, LYN, STOM, CD33, CCND3 (higher in ALL), CHRFAM7A, LEPROT, CAT and LTC4S.
  - Also above 0.9: ELANE (0.99), HOXA9 (0.99), SON (0.99, higher in ALL) and MYB (0.98, higher in ALL).
  - All of the listed probes are higher in AML unless marked otherwise.
- **Shrunken centroids** selected a median of 212 probes per fit; 8 probes were selected in at least 90% of subsamples.
  - The most frequent: CST3 0.99, CFD 0.95, CSTA 0.95, ELANE 0.92, CXCL8 0.92 (two probes), AZU1 0.91 and ZYX 0.90 (all higher in AML).
- **Original top-10 fold-change probes:**

| Probe | Symbol (GPL80) | Elastic net frequency | Shrunken centroids frequency |
|---|---|---|---|
| X82240_rna1_at | TCL1A | 0.08 | 0.72 |
| M84526_at | CFD | 0.37 | 0.95 |
| X76223_s_at | MAL | 0.00 | 0.38 |
| M27891_at | CST3 | 1.00 | 0.99 |
| M30703_s_at | AREG | 0.16 | 0.74 |
| L33930_s_at | CD24 | 0.00 | 0.58 |
| D88270_at | VPREB1 | 0.10 | 0.69 |
| M57731_s_at | CXCL2 | 0.13 | 0.85 |
| M20203_s_at | ELANE | 0.02 | 0.83 |
| M27783_s_at | ELANE | 0.99 | 0.92 |

- **Only two of the original top-10 were kept reliably by elastic net:** CST3 and one of the two ELANE probes. It selected the others in 37% of subsamples or fewer. Elastic net tends to keep one gene from a group of correlated genes, so a low frequency does not mean a gene carries no signal.
- **Shrunken centroids kept most of the original top-10 more often**, with frequencies of 0.38-0.99.
- **The stable genes are known lineage markers:** myeloid genes (CST3, CD33, ELANE, MPO, AZU1, CFD, LYN) are higher in AML, and lymphoid or cell-cycle genes (CCND3, MYB) are higher in ALL. This supports the biological reading in the report.

## 6. Bone marrow vs peripheral blood (test set)

`G_bone_marrow_vs_blood.csv`. The test set has 24 bone marrow samples (14 ALL, 10 AML) and 10 blood samples (6 ALL, 4 AML).

| Model (seed 2026) | Cut-off | BM errors / 24 | BM accuracy (95% CI) | PB errors / 10 | PB accuracy (95% CI) |
|---|---|---|---|---|---|
| KNN, honest | 0.5 | 1 | 0.958 (0.875-1.000) | 0 | 1.000 (1.000-1.000) |
| Elastic net | 0.5 | 2 | 0.917 (0.792-1.000) | 0 | 1.000 |
| Elastic net | Youden | 1 | 0.958 (0.875-1.000) | 0 | 1.000 |
| Shrunken centroids | 0.5 | 3 | 0.875 (0.750-1.000) | 0 | 1.000 |

- **At seed 2026, no model made an error on the 10 blood samples.** At seed 2028, each model made at most one blood error per cut-off. Four of those five errors were patient 67 (ALL, DFCI, false positive). The fifth was honest KNN at 0.5 missing patient 64 (AML, CCG).
- **A marrow-trained model did not do worse on blood, but this test is weak.** It rests on 4 blood AML and 6 blood ALL patients. A bootstrap interval of 1.000-1.000 only means there were no errors to resample, not that there is no uncertainty.
- **Hospital is mixed up with sample type.** The errors line up with the hospitals that have no training samples (section 3), and those hospitals contribute both marrow and blood samples. That confounder could matter more than sample type.

## 7. Seed sensitivity (2026 vs 2028)

The whole analysis was rerun with seed 2028, which changed the folds, bootstrap resamples, permutations and subsamples.

- **Unchanged:**
  - Elastic net CV AUC: 0.997 with both seeds.
  - Permutation p: 0.005 with both seeds.
  - Every test AUC: between 0.993 and 1.000.
  - Every CI still overlaps every other CI.
  - The tuned threshold raises KNN's CV sensitivity: 0.782 to 0.932 (seed 2026) and 0.759 to 0.941 (seed 2028).
  - The leaky vs honest pattern in sensitivity at 0.5: 0.936 vs 0.782 (seed 2026) and 0.950 vs 0.759 (seed 2028).
- **Changed, by one or two patients:**
  - Honest KNN chose k = 11 instead of k = 7, and caught 12/14 instead of 13/14 at 0.5.
  - Elastic net with the tuned cut-off caught 14/14 with 1 false positive, instead of 13/14 with none.
  - Shrunken centroids chose shrinkage 1.075 (886 genes) instead of no shrinkage, and caught 13/14 with 1 false positive, instead of 11/14 with none.
- **No conclusion changes.** Each of these changes is within the bootstrap CIs. That is itself a finding: the difference between models on this test set is the same size as the effect of the random seed.

## 8. How this relates to the original report

- **Problem 1, leaky CV:** confirmed. The CV AUC of 1.000 for every k came from leaky gene ranking and caret's tie-break. Honest nested CV gives AUC 0.969 for KNN.
- **Problem 3, threshold:** confirmed for KNN. A cut-off learned in training CV fixes most of the missed AML in CV. On the test set, the honest KNN already caught 13/14 at 0.5. Its final model chose k = 7, not k = 15, which is the most likely reason, but this was not tested separately.
- **Problem 6, panel size:** not re-tested as a curve. Where the training data chose the panel size themselves, they did not pick 2-5 genes: elastic net used 83 genes and shrunken centroids 886-3051. So the report's "3 genes is best" claim is not supported by any choice made without the test set. Treat it as a test-set observation, not a finding.
- **Blood samples:** the report says 9 of 10 blood samples were correct. The models here got 10 of 10 (seed 2026). Neither number can show that blood is as good as marrow, because there are only 10 blood samples and hospital is a confounder.

## 9. Leakage check: what was verified and how

1. **By design.** `fit_model()` receives only the fitting rows, `X[train, ]` and `y[train]`. Inside it, these steps are refitted from those rows only:
   - the variation filter;
   - fold-change ranking (KNN: redone in every inner fold as well);
   - z-scoring (KNN) or each package's internal standardisation;
   - the choice of k, lambda or shrinkage;
   - the Youden cut-off.

   Inner folds are drawn only from outer training rows. The test set appears only in `predict_model()`, after a model is fitted. All folds, permutations and subsamples are drawn in the main process before any fitting. The script asserts that every outer and inner fold contains both classes.
2. **Canary checks** (`I_leakage_canary_checks.csv`, all TRUE for all four models, both seeds):
   - The outer validation rows were replaced with random expression values and their labels flipped. The model was then refitted. Genes, k, lambda, shrinkage, cut-off and training-set predictions were all identical.
   - The final models were refitted with the test set replaced by noise and gave identical results.

   Because the fitting functions never receive those rows, these checks confirm the design rather than test something independent.
3. **Permutation null centred on 0.49**, which is what a leak-free pipeline should give (section 4).
4. **Information shared within the training data, disclosed:**
   - For elastic net and shrunken centroids, the variation filter and the lambda or shrinkage grid are computed on the outer training rows and shared by that fold's inner folds. The filter uses no labels.
   - Neither touches outer validation rows or the test set.

## 10. Caveats

- **Few AML in inner folds:** the inner fitting sets contain only about 7 AML patients. glmnet warned that a class had fewer than 8 observations. The run log reports 21 warnings in total; their individual text was not captured. In the smoke test they were this glmnet warning.
- **The bootstrap intervals for proportions are too narrow.** With 14 AML and 20 ALL, the percentile bootstrap cannot go below the observed value when there are no errors, and it understates the width otherwise. Exact Clopper-Pearson intervals are wider:

  | Result | Bootstrap (reported above) | Exact Clopper-Pearson |
  |---|---|---|
  | 13/14 AML recall | 0.786-1.000 | 0.661-0.998 |
  | 12/14 AML recall | 0.643-1.000 | 0.572-0.982 |
  | 20/20 specificity | 1.000-1.000 | 0.832-1.000 |
  | 4/4 blood AML | 1.000-1.000 | 0.398-1.000 |

  Use the exact intervals when quoting recall, specificity or accuracy. The bootstrap is kept for AUC, which has no exact interval.
- **Shrunken centroids posteriors are extreme.** On the test set, every posterior rounds to 0.000 or 1.000 (`D4_test_probabilities.csv`), so the missed AML are confidently wrong. This is why tuning the cut-off cannot change its results.
- **Ceiling effect:** performance is near the ceiling, and the test set has 14 AML. One patient changes recall by 0.071, so no difference between models in this appendix should be read as meaningful.

## sessionInfo()

```
R version 4.3.3 (2024-02-29)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 24.04.4 LTS

Matrix products: default
BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0

attached base packages:
[1] parallel  stats     graphics  grDevices utils     datasets  methods   base

other attached packages:
 [1] pROC_1.19.1         caret_6.0-94        lattice_0.22-5      ggplot2_3.4.4
 [5] pamr_1.57           glmnet_5.2          Matrix_1.6-5        golubEsets_1.44.0
 [9] Biobase_2.62.0      BiocGenerics_0.48.1
```

Full output, including locale and loaded namespaces: `results/sessionInfo.txt`.
