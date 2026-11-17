# Appendix: more rigorous machine learning on the Golub ALL vs AML data

This appendix re-examines the group's Golub (1999) analysis with stricter methods, and explains what each method is and why it was used. Every number comes from one run of `appendix_advanced_ml.R` (version 2). Main results use seed 2026 (`results/`); the same analysis rerun with seed 2028 is in `results/seed2028/`.

Each analysis section follows the same pattern:

- **What:** what was done.
- **Why:** the problem it addresses.
- **Result:** the numbers.
- **Meaning:** what the group can conclude.

Section 1 explains the terms used throughout.

---

## 1. Key terms

### Measures of performance

AML is the "positive" class throughout, because missing an AML patient is the more dangerous error: they would receive the wrong chemotherapy.

| Term | Plain meaning |
|---|---|
| **Sensitivity (AML recall)** | Of all true AML patients, the share the model calls AML. 13/14 = 0.929. |
| **Specificity** | Of all true ALL patients, the share the model calls ALL. |
| **Accuracy** | Share of all patients classified correctly. With 27 ALL and 11 AML in training, a model that always says "ALL" already scores 71%, so accuracy alone can mislead. |
| **Balanced accuracy** | The average of sensitivity and specificity. The "always ALL" model scores 0.5 on it, so the class imbalance cannot hide a poor model. |
| **AUC** (area under the ROC curve) | The chance that a randomly chosen AML patient receives a higher AML score than a randomly chosen ALL patient. 0.5 = guessing, 1.0 = perfect ranking. AUC measures how well the model *ranks* patients and does not depend on any cut-off. |
| **Cut-off (decision threshold)** | Each model gives a score, P(AML), between 0 and 1. A patient is called AML if the score is at or above the cut-off. The default is 0.5. |

### Youden's index and the Youden cut-off

- **What it is.** Youden's J = sensitivity + specificity − 1. It is 0 for a useless test and 1 for a perfect one. The **Youden cut-off** is the value of the cut-off that makes J as large as possible: the point where catching AML and correctly clearing ALL are best balanced, with both errors counted as equally bad.
- **Why we used it.** The original model had AUC 0.991, meaning it ranked almost every AML patient above almost every ALL patient, yet it missed 5 of 14 AML at the 0.5 cut-off.
  - The missed AML patients scored 0.12-0.40: ranked correctly, but under 0.5.
  - The cause is imbalance. With 27 ALL and 11 AML in training, most of a patient's k nearest neighbours tend to be ALL, so AML scores are pulled down.
  - The fix is to move the cut-off, not to change the model. Youden's J is the standard, assumption-free way of choosing where to move it.
- **The rule that makes it valid.** The cut-off must be chosen using training patients only, from predictions made on patients the model had not been fitted to (inner-fold out-of-fold predictions, explained below).
  - Choosing it on the patients being scored would make the result look better than it is. This is "leakage".
  - When several cut-offs tie on J, we take the one closest to 0.5, so the default is kept unless the data clearly argue against it.

### Cross-validation

| Term | Plain meaning |
|---|---|
| **Cross-validation (CV)** | Split the training patients into 5 groups ("folds"). Fit on 4 folds, predict the 5th, and rotate until every patient has been predicted once by a model that never saw them. |
| **Stratified** | Each fold keeps roughly the same ALL:AML ratio as the whole set, so no fold is all ALL. The script checks that every fold contains both classes. |
| **Repeated** | The random split into folds is redone several times and the results averaged, so the answer does not depend on one lucky or unlucky split. |
| **Hyperparameter** | A setting chosen before fitting rather than learned from the data, for example k in KNN. |
| **Nested CV** | CV inside CV. The *outer* loop scores the method. The *inner* loop, run separately inside each outer training part, chooses the hyperparameters and the cut-off. A score is only honest if nothing about the patients being scored was used to make choices. |
| **Data leakage** | Any route by which the patients being scored influence the model. The original script ranked genes using all 38 training patients *before* CV, so each CV fold was scored with genes chosen partly on that fold's own patients. |
| **Range across repeats** | The CV figures in brackets are the 2.5th-97.5th percentiles over the 20 outer repeats. They show how much the result moves when the same 38 patients are re-split. **They are not confidence intervals**, because they say nothing about new patients. |

### Models

| Term | Plain meaning |
|---|---|
| **KNN** | k-nearest neighbours. A new patient's AML score is the share of AML among the k most similar training patients, with similarity measured as Euclidean distance on z-scored genes. |
| **\|log2 fold change\| ranking** | Genes ranked by how different the two class means are. Used in the original script. It ignores how much patients vary within each class. |
| **t-statistic ranking** | Genes ranked by the difference in means divided by its standard error. A gene with a big difference but large spread within classes ranks lower. Golub et al. used a similar signal-to-noise score. |
| **Elastic net** | Logistic regression with a penalty that shrinks most gene weights to exactly zero, so the model chooses its own genes. It mixes the "lasso" and "ridge" penalties (alpha = 0.5). **lambda** sets how strong the penalty is. **Class weights** make the 11 AML count as much in total as the 27 ALL. |
| **Nearest shrunken centroids** (pamr) | Computes each class's average profile (its centroid) and shrinks each gene's class difference towards zero. Genes that shrink to zero drop out. The **shrinkage threshold** sets how hard it shrinks. **Equal priors** stop the larger ALL group being favoured. |
| **Minimum-error rule vs one-standard-error (1-SE) rule** | Two ways of choosing the shrinkage. Minimum-error takes the setting with the lowest CV error. 1-SE takes the strongest shrinkage, and so the fewest genes, whose error is within one standard error of the minimum. The 1-SE rule trades a tiny loss in CV accuracy for a simpler, usually more stable model. |

### Uncertainty and significance

| Term | Plain meaning |
|---|---|
| **95% confidence interval (CI)** | A range that, under repeated sampling, would contain the true value 95% of the time. With 14 AML patients, intervals are wide. |
| **Exact (Clopper-Pearson) CI** | The standard interval for a proportion such as 13/14. It is valid even with very few patients and even when there are no errors. |
| **Bootstrap CI** | Resample the test patients with replacement 2000 times, recompute the measure each time, and take the middle 95%. Used here only for AUC, which has no simple exact interval. |
| **Permutation test** | Shuffle the ALL/AML labels at random and rerun the entire analysis. If the real labels give a far better score than any shuffled labels, the signal is real, not an artefact of the method. |
| **p-value** | The chance of a result at least this extreme if there were no real signal. |
| **Selection frequency** | Refit the model on many random 80% subsets of the training patients and count how often each gene is chosen. Genes chosen nearly every time are robust. |

---

## 2. Starting point: reproducing the original analysis

- **What:** we re-ran the original method exactly. Genes were ranked on all 38 training patients, then caret's 5-fold CV chose k from 1 to 15 (`A1_original_caret_leaky_cv.csv`).
- **Why:** to make sure the appendix starts from the same place as the report before changing anything.
- **Result:**
  - The top-10 genes matched the report exactly.
  - 3051 genes passed the variation filter.
  - CV AUC was 1.000 for every k, so caret's tie-break picked k = 15.
- **Meaning:** the starting point is correct. A CV AUC of exactly 1.000 at every k is itself a warning sign. CV could not tell the k values apart, so k = 15 was chosen by a default rule, not by the data.

Gene symbols come from the NCBI GEO annotation of the Hu6800 array (GPL80), because the Bioconductor annotation package was not installed. In that annotation, probe D88270_at is **VPREB1**; the report's Table 2 calls it IGL.

---

## 3. Nested cross-validation: how well does each method really do?

**What.** Seven procedures were run through the same nested CV, using identical folds:

- **Outer loop:** stratified 5-fold, repeated 20 times. Every training patient is scored 20 times by a model that never saw them.
- **Inner loop:** stratified 5-fold, repeated 10 times, inside every outer training part. It chooses k, the number of genes, lambda or shrinkage, and the Youden cut-off.
- **Refitted inside every fold:** every data-dependent step, including the variation filter, gene ranking, z-scoring, the hyperparameter and the cut-off.

**Why each procedure was included.**

| Procedure | Why |
|---|---|
| KNN, leaky | The original method (genes ranked once on all 38), inside the new framework, for comparison. |
| KNN, honest, \|log2 FC\| | The original method done properly: genes re-ranked inside every fold. Answers known problem 1. |
| KNN, honest, t-statistic | Same, but ranking genes by t-statistic. Answers known problem 2 (fold change ignores within-class variance). |
| KNN, honest, panel size tuned | The inner CV chooses the panel size (1, 2, 3, 5, 10, 20 or 50 genes) as well as k. Answers known problem 6 (panel size was read off the test set). |
| Elastic net | Pre-declared primary model: a different kind of model that picks its own genes and handles imbalance with class weights. Answers known problem 5 (only one model was tried). |
| Shrunken centroids, minimum-error rule | A third, very different model designed for gene-expression data. Pre-declared. |
| Shrunken centroids, 1-SE rule | Added **after** seeing the minimum-error rule choose no shrinkage on a one-patient difference. Post hoc, so it is labelled as such and not used for any headline claim. |

Version 2 repeats the inner CV 10 times. In version 1 a single inner split was used, and the chosen settings jumped around between seeds (for example k = 7 vs k = 11). Averaging over 10 inner splits makes the choices stable: honest KNN now chooses k = 11 with both seeds.

**Result** (`A2_nested_cv_performance.csv`, seed 2026). Figures are the mean, with the range across repeats in brackets.

| Procedure | AUC | Sensitivity at 0.5 | Sensitivity at Youden | Balanced accuracy at 0.5 | Balanced accuracy at Youden |
|---|---|---|---|---|---|
| KNN, leaky | 0.982 (0.953-1.000) | 0.927 | 0.959 | 0.964 | 0.980 |
| KNN, honest, \|log2 FC\| | 0.992 (0.954-1.000) | **0.709** | 0.968 | 0.854 | 0.973 |
| KNN, honest, t-statistic | **1.000** (0.997-1.000) | 0.973 | 0.986 | **0.981** | 0.984 |
| KNN, honest, panel tuned | 0.967 (0.876-1.000) | 0.718 | 0.923 | 0.858 | 0.950 |
| Elastic net (primary) | 0.996 (0.987-1.000) | 0.964 | 0.986 | 0.969 | 0.979 |
| Shrunken centroids, minimum-error | 0.993 (0.967-1.000) | 0.941 | 0.941 | 0.956 | 0.957 |
| Shrunken centroids, 1-SE (post hoc) | 0.983 (0.958-1.000) | 0.932 | 0.914 | 0.931 | 0.928 |

Specificity was 0.93-1.00 for every procedure (`A2` gives the full table).

Tuning behaviour (`A3_nested_cv_tuning.csv`):

- **Leaky KNN** had tied choices in 87 of 100 outer fits, and chose k = 1 or 3 in 69 of them.
- **Honest \|log2 FC\| KNN** mostly chose k = 9 or 11 (47 fits).
- **The panel-tuned KNN** chose 50 genes in 87 of 100 fits, 20 genes in 7, 10 in 4 and 5 in 2. It never chose 1, 2 or 3.
- **Elastic net** used 68-94 genes (median 80).

**Meaning.**

- **The leak changed which k was chosen, not mainly the AUC.**
  - With leaked genes, the training patients separate perfectly, many k values tie, and the tie-break falls to small k.
  - With honest ranking, CV prefers larger k (9-11). With 27 ALL vs 11 AML, large k dilutes AML votes, so AML recall at 0.5 falls to 0.709.
  - This is exactly the problem the original report found on the test set (9/14 AML). The original's CV AUC of 1.000 hid it.
  - The Youden cut-off fixes most of it: sensitivity rises from 0.709 to 0.968.
- **Ranking genes by t-statistic worked better than fold change** (known problem 2). It gave AUC 1.000 vs 0.992, and balanced accuracy at 0.5 of 0.981 vs 0.854. Seed 2028 agreed: 0.999 vs 0.991, and 0.990 vs 0.832. Genes that differ consistently, not just on average, gave neighbours that vote correctly even at the default cut-off.
- **When the data choose the panel size, they choose large panels** (known problem 6). Tuning the size cost a little AUC (0.967 vs 0.992 for a fixed 50), because the noisy inner CV occasionally picked a smaller panel.
- **Elastic net was the most stable** procedure (AUC range 0.987-1.000).
- **The post-hoc 1-SE rule did not improve shrunken centroids in CV** (AUC 0.983 vs 0.993).
- **Pooling vs averaging folds makes no difference** (`A4_pooled_vs_fold_averaged_auc.csv`): pooled and fold-averaged AUC agree to within 0.008 for every procedure. So the original 1.000 was not caused by averaging over small folds.

---

## 4. The decision threshold: 0.5 vs Youden

- **What:** each procedure was scored at the default 0.5 cut-off and at its Youden cut-off. The Youden cut-off was chosen from inner out-of-fold predictions on training patients only, averaged over the 10 inner repeats.
- **Why:** known problem 3. The original model ranked patients almost perfectly (AUC 0.991) but missed 5 AML at 0.5, which points to the cut-off, not the model.
- **Result in CV** (section 3 table):
  - **KNN:** this is where the threshold matters. Sensitivity rose from 0.709 to 0.968 for \|log2 FC\| KNN (seed 2028: 0.668 to 0.964), and from 0.718 to 0.923 for panel-tuned KNN. The median tuned cut-off for \|log2 FC\| KNN was 0.217, far below 0.5.
  - **Elastic net:** it was already trained with class weights, which correct the imbalance inside the model. Tuning changed little (0.964 to 0.986; seed 2028: 0.973 to 0.964).
  - **Shrunken centroids:** their posteriors are almost always 0 or 1, so moving the cut-off changes nothing.
- **Result on the test set** (section 5): the tuned cut-off added one AML for \|log2 FC\| KNN and panel-tuned KNN (12 to 13 of 14), at the cost of one false positive. It changed nothing for the other models at seed 2026.
- **Meaning:** the report's explanation is right. The missed AML were a cut-off problem caused by imbalance, and a cut-off learned from training data fixes it for KNN. For a model that already weights the classes, such as elastic net, a separate threshold step adds little.

---

## 5. The independent test set (34 patients, used once)

**What.** Each procedure was refitted on all 38 training patients exactly as in CV, including the 10 x 5-fold inner CV for its settings and cut-off. It then predicted the 34 test patients **once**. Nothing — no model, gene number, setting or cut-off — was chosen using the test set.

**Why.** CV estimates come from the same 38 patients. The test set comes from a separate set of patients, partly from other hospitals, and is the fairest check.

**Uncertainty.** Recall, specificity and accuracy use exact Clopper-Pearson CIs. Version 1 used the bootstrap, which was too narrow here: for 20/20 it gave 1.000-1.000 instead of 0.832-1.000. AUC uses a bootstrap CI.

**Result** (`D1`, `D2`, `D3`, seed 2026):

| Procedure | Final setting | Test AUC (95% CI) | AML recall at 0.5 (exact CI) | At Youden cut-off |
|---|---|---|---|---|
| Original report (for reference) | k = 15, 50 genes | 0.991 | 9/14 = 0.643 (0.351-0.872) | n/a |
| KNN, leaky | k = 5, 50 genes | 0.998 (0.989-1.000) | 13/14 = 0.929 (0.661-0.998) | 13/14, 0 FP (cut 0.260) |
| KNN, honest, \|log2 FC\| | k = 11, 50 genes | 0.996 (0.979-1.000) | 12/14 = 0.857 (0.572-0.982) | 13/14, 1 FP (cut 0.205) |
| KNN, honest, t-statistic | k = 9, 50 genes | 0.996 (0.979-1.000) | 13/14, 1 FP | 13/14, 1 FP (cut 0.556) |
| KNN, honest, panel tuned | k = 11, 50 genes | 0.996 (0.979-1.000) | 12/14 | 13/14, 1 FP (cut 0.205) |
| Elastic net (primary) | 83 genes | 0.996 (0.979-1.000) | 12/14 | 12/14, 0 FP (cut 0.566) |
| Shrunken centroids, minimum-error | 3051 genes (no shrinkage) | 0.993 (0.968-1.000) | 11/14 = 0.786 (0.492-0.953) | 11/14, 0 FP |
| Shrunken centroids, 1-SE (post hoc) | 1201 genes | 0.993 (0.968-1.000) | 13/14, 1 FP | 13/14, 1 FP |

Notes:

- **FP** is a false positive: an ALL patient called AML. With no FP, specificity is 20/20 = 1.000 (0.832-1.000); with one FP, 19/20 = 0.950 (0.751-0.999).
- **Test accuracy** was 31-33 of 34 for every procedure. For example, 32/34 = 0.941 (0.803-0.993).

**Meaning.**

- **No procedure can be declared best.** Every test AUC is 0.993-0.998 with overlapping intervals, and every AML recall interval overlaps every other.
- **Every honest procedure caught more AML at 0.5 than the original (11-13 vs 9 of 14).** The most likely reason is that the original's k = 15 was chosen by a tie-break, not by the data.
- **The "leaky" row is not leaky on the test set.** Its final model is trained on all 38 patients like the others. It differs only in which k and cut-off its inner CV chose.

**Who was misclassified** (`D5_misclassified_test_patients.csv`):

- **Patient 66** (AML, bone marrow, CCG) was missed by **every procedure at both cut-offs** with seed 2026. With seed 2028 it was missed by every procedure at 0.5 and by all but leaky KNN at the Youden cut-off. Elastic net scored this patient 0.307.
- **Other missed AML:** patient 60 (bone marrow, St-Jude: elastic net and minimum-error centroids), patient 54 (bone marrow, St-Jude: minimum-error centroids), and patient 64 (blood, CCG: \|log2 FC\| and panel KNN at 0.5).
- **Every missed AML patient came from a hospital not represented in training** (St-Jude or CCG).
- **The only false positive was patient 67** (ALL, blood, DFCI), for the t-statistic KNN, both KNN variants at their Youden cut-offs, and 1-SE centroids.

---

## 6. Hospital: a confounder found during the review

**What.** The script tabulates the hospital each sample came from (`J1`, `J2`). It then compares test scores and errors between hospitals that were and were not represented in training (`J3`, Figure 7).

**Why.** In the training set, hospital and diagnosis turn out to be identical:

| Training set | ALL | AML |
|---|---|---|
| DFCI | 27 | 0 |
| CALGB | 0 | 11 |

So any difference in how the two hospitals collected or processed samples is perfectly mixed with the ALL/AML difference. From the training data alone, nobody can tell whether a model learned "AML" or "processed at CALGB".

**Result.**

- **The test set contains two new hospitals:** St-Jude (3 ALL, 5 AML) and CCG (5 AML).
- **The model is not detecting hospital alone.** St-Jude's 3 ALL were still scored as clearly ALL: the highest elastic net score was 0.014.
- **But AML from the new hospitals scored lower than AML from CALGB:**
  - elastic net: median 0.839 vs 0.994 (Wilcoxon p = 0.036);
  - \|log2 FC\| KNN: 0.591 vs 0.864 (p = 0.022).
  - For the t-statistic KNN and the centroid models, scores were mostly exactly 1.000, so no difference could show (p = 0.18-0.28).
- **Errors at 0.5:** 1-3 of 13 patients from new hospitals vs 0-1 of 21 from training hospitals. Only one procedure reached p < 0.05 (minimum-error centroids, p = 0.048).

**Meaning.**

- The ALL/AML signal is real.
- Part of what the models learned may be specific to the AML samples from CALGB, which is the most likely reason every missed AML came from a new hospital.
- This should be stated as a limitation in both the main report and the appendix.

---

## 7. Bone marrow vs peripheral blood

- **What:** test performance was split by sample source (`G_bone_marrow_vs_blood.csv`). The test set has 24 bone marrow samples (14 ALL, 10 AML) and 10 blood samples (6 ALL, 4 AML). All training samples are bone marrow.
- **Why:** question 3 of the report. Does a model trained on marrow work on blood, which is much less invasive to collect?
- **Result** (seed 2026, errors / samples):

| Procedure | Bone marrow, 0.5 | Blood, 0.5 | Blood, Youden |
|---|---|---|---|
| KNN, honest, \|log2 FC\| | 1/24 | 1/10 (missed AML 64) | 1/10 (false positive 67) |
| KNN, honest, t-statistic | 1/24 | 1/10 (67) | 1/10 (67) |
| Elastic net | 2/24 | 0/10 | 0/10 |
| Shrunken centroids, minimum-error | 3/24 | 0/10 | 0/10 |

  - With 4 blood AML, recall of 4/4 still has an exact CI of 0.398-1.000.
  - With 6 blood ALL, specificity of 6/6 has 0.541-1.000.
- **Meaning:** there was no sign that blood is worse than marrow. However, 10 blood samples cannot show that it is as good. In the test set, sample type is also mixed with hospital: all CCG blood samples are AML and all DFCI blood samples are ALL.

---

## 8. How many genes are needed? (panel-size curve)

**What.** KNN with \|log2 FC\| ranking was run as seven fixed procedures (1, 2, 3, 5, 10, 20 and 50 genes), each through the full nested CV with k and the cut-off tuned inside (`K_panel_size_curve.csv`, Figure 6). Each fixed panel was also applied to the test set, for description only; nothing was chosen from the test results.

**Why.** Known problem 6: the report's "3 genes is best" was read off the 34 test patients, so the test set was effectively used to choose.

**Result** (seed 2026; CV figures are means over repeats):

| Genes | CV AUC | CV balanced accuracy (Youden) | Test accuracy at 0.5 (exact CI) |
|---|---|---|---|
| 1 | 0.754 | 0.698 | 0.588 (0.407-0.754): every patient called ALL |
| 2 | 0.859 | 0.812 | 0.971 (0.847-0.999) |
| 3 | 0.918 | 0.825 | 0.971 (0.847-0.999) |
| 5 | 0.968 | 0.908 | 0.941 (0.803-0.993) |
| 10 | 0.961 | 0.921 | 0.912 (0.763-0.981) |
| 20 | 0.955 | 0.934 | 0.941 (0.803-0.993) |
| 50 | 0.992 | 0.973 | 0.941 (0.803-0.993) |

Seed 2028 showed the same pattern: CV AUC 0.719 with 1 gene, 0.934 with 3, 0.963 with 5 and 0.991 with 50.

**Meaning.**

- **The training data say performance keeps improving up to 50 genes,** and that at least 5 genes are needed for a CV AUC above 0.95.
- **The test set again favours 2-3 genes (33/34),** reproducing the report's pattern. But its intervals overlap those for 50 genes, and the training data do not support it. This is the noise of 34 patients.
- **The single-gene panel is fragile:** on the test set it called every patient ALL at 0.5 (AUC 0.911).
- **Recommendation:** report "about 5 or more genes; 50 was best in CV", not "3 genes is enough".

---

## 9. Permutation test: is the signal real?

**What.** For the primary model (elastic net), the training labels were shuffled at random 200 times. Each time, the **entire** nested CV was rerun (`E2`, Figure 4).

**Why.** Known problem 4. A complicated pipeline can sometimes produce good-looking numbers from pure noise, especially if something leaks. If shuffled labels also score well, the result means nothing.

**Result:**

| Seed | Observed mean CV AUC | Shuffled labels: mean (SD) | 95th percentile | Maximum | p |
|---|---|---|---|---|---|
| 2026 | 0.996 | 0.504 (0.072) | 0.684 | 0.821 | 0.005 |
| 2028 | 0.997 | 0.491 (0.057) | 0.619 | 0.784 | 0.005 |

**Meaning.**

- With shuffled labels the pipeline scores about 0.5, which is chance, as it should. That is also evidence of no leakage.
- No shuffle came close to the real 0.996. p = 1/201 is the smallest value possible with 200 shuffles.
- The ALL/AML signal is real. For this well-known dataset that was expected, so the test mainly works as a check on the pipeline.

---

## 10. Which genes does the model rely on? (gene stability)

**What.** Elastic net and both shrunken-centroid procedures were refitted on 100 random stratified 80% subsets of the training patients, with full inner CV each time. We recorded how often each probe was selected (`F_gene_selection_frequency.csv`, `F_original_top10_selection.csv`, Figure 5).

**Why.** A gene list from one fit can be an accident of which patients happened to be included. Genes chosen in nearly every subset are the robust ones, and they can be checked against known biology.

**Result** (seed 2026):

- **Elastic net** used a median of 82 probes per fit; 26 probes were chosen in at least 90% of subsets.
  - Chosen every time (among others): CST3, ZYX, RBBP4 (higher in ALL), LYN, STOM, CD33, CHRFAM7A, LEPROT, HOXA9 and LTC4S. All of these are higher in AML except RBBP4.
- **Shrunken centroids, minimum-error rule:** a median of 1991 probes per fit (367 chosen in at least 90% of subsets). It keeps almost every gene, so its frequencies say little.
- **Shrunken centroids, 1-SE rule:** a median of 236 probes per fit (32 chosen in at least 90%). The most frequent were CST3 1.00, CFD 0.99, ELANE 0.99, CXCL8 0.99 (two probes), CSTA 0.99, ZYX 0.99, MGST1 0.97, MPO 0.96 and AZU1 0.95, all higher in AML.
- **The original top-10 fold-change genes:**

| Probe | Symbol | Elastic net | Centroids, 1-SE |
|---|---|---|---|
| X82240_rna1_at | TCL1A | 0.08 | 0.86 |
| M84526_at | CFD | 0.39 | 0.99 |
| X76223_s_at | MAL | 0.00 | 0.33 |
| M27891_at | CST3 | 1.00 | 1.00 |
| M30703_s_at | AREG | 0.10 | 0.87 |
| L33930_s_at | CD24 | 0.00 | 0.64 |
| D88270_at | VPREB1 | 0.07 | 0.77 |
| M57731_s_at | CXCL2 | 0.05 | 0.93 |
| M20203_s_at | ELANE | 0.01 | 0.91 |
| M27783_s_at | ELANE | 0.96 | 0.99 |

**Meaning.**

- **The robust genes are known lineage markers:** myeloid genes (CST3, CD33, MPO, ELANE, AZU1, CFD, LYN) are higher in AML. This supports the report's biological reading.
- **Elastic net keeps only CST3 and one ELANE probe from the original top-10.** It tends to pick one gene from a group of genes that carry the same information, so a low frequency (for example TCL1A 0.08) means "a correlated gene was chosen instead", not "uninformative".
- **The 1-SE centroid model keeps most of the original top-10** (0.33-1.00).
- **Frequencies are somewhat optimistic,** because any two 80% subsets share most of their patients.

---

## 11. Does anything depend on the random seed?

- **What:** the whole analysis was rerun with seed 2028, which changed every fold, permutation, subsample and bootstrap resample.
- **Why:** to check that no conclusion rests on a lucky random split. The original report also re-ran with seed 2028.
- **Result:**

| Quantity | Seed 2026 | Seed 2028 |
|---|---|---|
| Elastic net CV AUC | 0.996 | 0.997 |
| t-statistic KNN CV AUC | 1.000 | 0.999 |
| \|log2 FC\| KNN CV sensitivity, 0.5 to Youden | 0.709 to 0.968 | 0.668 to 0.964 |
| Permutation p | 0.005 | 0.005 |
| Final \|log2 FC\| KNN k | 11 | 11 |
| Elastic net test AML recall at 0.5 | 12/14 | 12/14 |
| Elastic net test recall at Youden | 12/14 (cut 0.566) | 13/14, 1 FP (cut 0.346) |
| Minimum-error centroids final genes | 3051 | 1582 |
| Minimum-error centroids test recall | 11/14 | 13/14, 1 FP |
| 1-SE centroids final genes | 1201 | 638 |

- **Meaning:**
  - No conclusion changes.
  - Repeating the inner CV made the KNN and elastic net choices stable.
  - The shrunken-centroid gene count is still unstable, so treat that model as the least reliable.
  - Test counts move by at most two patients, which is within every interval.

---

## 12. Leakage check

**What and how it was verified:**

1. **By design.**
   - Every fitting step gets only its fitting rows. These steps are refitted inside each fold: the variation filter, gene ranking (KNN: in every inner fold too), z-scoring or package standardisation, the hyperparameter and the cut-off.
   - Inner folds are drawn only from outer training rows.
   - The test set is used only for prediction.
   - All random splits are drawn before any fitting, and every fold is checked to contain both classes.
2. **Canary checks** (`I_leakage_canary_checks.csv`: all TRUE for all procedures and both seeds).
   - The outer validation patients were replaced with random noise and their labels flipped. The models were then refitted, and every fitted quantity was identical.
   - Because the fitting code never receives those rows, this confirms the design rather than testing it independently.
3. **Permutation null at about 0.5** (section 9), as expected without leakage.
4. **Information shared within the training data.** The variation filter and the elastic net lambda sequence are computed on the outer training rows and shared by their inner folds. The filter uses no labels, and neither touches validation or test patients.

---

## 13. What changed from version 1, and why

| Change | Reason |
|---|---|
| Inner CV repeated 10 times | Version 1's single inner split made k, lambda and shrinkage depend on the seed. |
| Exact Clopper-Pearson CIs for test proportions | The bootstrap was too narrow with 14 AML (for example 1.000-1.000 for 20/20). |
| CV brackets labelled "range across repeats" | They are not confidence intervals. |
| KNN with t-statistic ranking | Known problem 2. |
| KNN with tuned panel size, and the panel-size curve | Known problem 6. |
| Shrunken centroids with the 1-SE rule | The minimum-error rule picked no shrinkage on a one-patient difference (post hoc, labelled). |
| Hospital tables and Figure 7 | The training hospital equals the diagnosis. |

Some version 1 numbers changed as a result. For example, honest \|log2 FC\| KNN CV AUC rose from 0.969 to 0.992, because version 1's unstable choice of k pulled it down. Version 2 supersedes version 1; the earlier numbers remain in the git history.

---

## 14. What the report can and cannot claim

**Can claim:**

- Honest nested CV confirms very strong ALL vs AML separation: CV AUC 0.967-1.000 across procedures, 0.996 for the primary elastic net.
- The original CV AUC of 1.000 came from ranking genes before CV. It hid the real weakness: low AML recall at 0.5 with a large k.
- A cut-off learned from training data (Youden) fixes most missed AML for KNN.
- Ranking genes by t-statistic works better than fold change in CV.
- All seven procedures agree on the test set (AUC 0.993-0.998), and every honest procedure beat the original's 9/14 AML recall at 0.5.
- The genes the models rely on are known myeloid and lymphoid markers.

**Cannot claim:**

- That any one model is best.
- That 2-3 genes are enough. The training data say about 5 or more, with 50 best.
- That blood works as well as marrow. There are 10 blood samples, and hospital is mixed with sample type.
- That the model works equally well at a new hospital. The training hospital equals the diagnosis, and every missed AML came from a new hospital.

---

## 15. Caveats

- The test set has 14 AML patients, so one patient changes recall by 0.071.
- Inner fitting sets hold about 7 AML. glmnet warned that a class had fewer than 8 observations; these warnings appear throughout the run log.
- Shrunken-centroid scores are almost always exactly 0 or 1, so its missed AML are confidently wrong, and AUC and cut-off tuning say little about it.

---

## sessionInfo()

```
R version 4.3.3 (2024-02-29)
Platform: x86_64-pc-linux-gnu (64-bit)
Running under: Ubuntu 24.04.4 LTS

other attached packages:
 [1] pROC_1.19.1         caret_6.0-94        lattice_0.22-5      ggplot2_3.4.4
 [5] pamr_1.57           glmnet_5.2          Matrix_1.6-5        golubEsets_1.44.0
 [9] Biobase_2.62.0      BiocGenerics_0.48.1
```

Full output: `results/sessionInfo.txt`.
