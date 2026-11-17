# Review of the appendix results

This review checks the appendix analysis (`appendix_advanced_ml.R`, `RESULTS.md`, `results/`) for errors, weak points and overclaims. The aim is to show what the group can safely say in the report and what it cannot. Where a point needed a new calculation, the numbers come from the saved results or from the Golub clinical data (`pData`); nothing was rerun with different settings.

## Verdict

The main conclusions hold:

- The original CV estimate was leaky.
- All three honest models separate ALL from AML very well.
- No model can be ranked above another on 34 test patients.
- The signal is real (permutation p = 0.005).

The review found three problems that matter more than anything in the original brief:

1. **In the training set, hospital and diagnosis are the same thing.** Every ALL comes from one hospital and every AML from another. This affects the whole project, the original report included.
2. **The uncertainty is understated in two places.** The bootstrap intervals for proportions are too narrow, and the CV ranges are not confidence intervals.
3. **RESULTS.md contained three factual errors about individual patients and one unsupported causal claim.** All are now corrected (section 5).

## 1. What holds up

- **The leakage control is sound.**
  - Every fitting step gets only its fitting rows, and the inner folds sit inside the outer training part.
  - The test set is used only for prediction.
  - The permutation null is centred at 0.49, as a leak-free pipeline should be.
  - The script reproduces the original analysis exactly (same top-10 genes, CV AUC 1.000 for every k, k = 15), so the comparison starts from the right place.
- **Conclusions do not depend on the seed.** Elastic net CV AUC was 0.997 with both seeds, and permutation p was 0.005 with both. The leaky vs honest sensitivity gap repeats: 0.936 vs 0.782, then 0.950 vs 0.759.
- **The threshold result is solid for KNN.** CV sensitivity rose from 0.782 to 0.932 (seed 2028: 0.759 to 0.941) when the cut-off was learned in training CV. This supports the report's explanation that the missed AML came from class imbalance.
- **The stable genes make biological sense.** CST3, CD33, ELANE, MPO, AZU1 and LYN are higher in AML; CCND3 and MYB are higher in ALL.

## 2. Major issues

### 2.1 Hospital and diagnosis are confounded in the training set (new finding)

From `pData(Golub_Train)`:

| Training set | ALL | AML |
|---|---|---|
| DFCI | 27 | 0 |
| CALGB | 0 | 11 |

- **The problem:** any difference in how the two hospitals collected, stored or processed samples is perfectly mixed with the ALL/AML difference. A model trained on these 38 patients cannot know which of the two it is learning.
- **The test set gives a partial check,** because two new hospitals appear (St-Jude: 3 ALL, 5 AML; CCG: 5 AML):
  - St-Jude's 3 ALL were still scored as clearly ALL by elastic net (0.001, 0.004, 0.014), so the model is not simply detecting "DFCI vs CALGB".
  - However, AML from the new hospitals were scored lower than AML from CALGB: elastic net median 0.84 vs 0.99, Wilcoxon p = 0.034.
  - Every misclassified patient at seed 2026 came from a new hospital: 3 of 13 vs 0 of 21, Fisher p = 0.048.
  - Restricted to AML only, the difference is not significant (3 of 10 vs 0 of 4, p = 0.51), because there are only 4 CALGB AML in the test set.
- **What this means for the report:**
  - The ALL/AML signal is real, but part of what the model learned may be specific to CALGB. That is the most likely reason the misses are AML from new hospitals.
  - It also undermines the blood vs marrow question. Hospital is mixed with sample type in the test set too: all CCG blood samples are AML, and all DFCI blood samples are ALL.
  - This should go in the limitations of both the main report and the appendix.

### 2.2 The intervals understate the uncertainty

- **Bootstrap intervals for proportions are too narrow.** With few patients, the percentile bootstrap cannot fall below the observed value when there are no errors, and it is narrow otherwise. Exact Clopper-Pearson intervals:

  | Result | Bootstrap (in D2) | Exact |
  |---|---|---|
  | 13/14 AML recall | 0.786-1.000 | 0.661-0.998 |
  | 12/14 | 0.643-1.000 | 0.572-0.982 |
  | 11/14 | 0.571-1.000 | 0.492-0.953 |
  | 20/20 specificity | 1.000-1.000 | 0.832-1.000 |
  | 4/4 blood AML | 1.000-1.000 | 0.398-1.000 |
  | 32/34 accuracy | 0.853-1.000 | 0.803-0.993 |

  For example, the honest models' AML recall of 11-13 of 14 is compatible with anything from about 50% to 100%.
- **The CV ranges are not confidence intervals.** The "(2.5th-97.5th percentile over 20 repeats)" ranges measure only how much the result moves when the same 38 patients are re-split. They say nothing about how the result would change with new patients.
  - The repeats are highly correlated. Elastic net's 20 repeat AUCs take only four distinct values (0.990, 0.993, 0.997, 1.000).
  - RESULTS.md labels them correctly, but readers may take them as CIs. In the report, call them "range across repeats".

### 2.3 The final models are not fixed objects

- **The test results depend on one inner split.** Each final model is chosen by a single stratified 5-fold inner split of the 38 patients. Changing the seed changed that split, and with it:
  - honest KNN's k (7 to 11);
  - shrunken centroids' shrinkage (0 to 1.075, i.e. 3051 to 886 genes);
  - several test counts, by one or two patients.
- **So the test set evaluates the procedure, not one model.** The fair statement is that the procedure gives test recall of roughly 12-14 of 14 depending on the split. Repeating the inner CV several times and averaging (or bagging the final model) would make the final model stable.

### 2.4 The leaky vs honest comparison is partly shaped by the tie-break rule

- **Only the gene ranking leaks.** In the appendix's "leaky KNN", z-scoring and k are still handled inside the folds. That is a clean test of one leak, but it is not the full original pipeline.
- **The tie-break pushes leaky KNN to k = 1.** With leaked genes, many k values tie on inner AUC. The pre-declared tie-break (lowest Brier score, then smallest k) then picks k = 1 or 3 in 90 of 100 fits.
  - A k = 1 model gives only 0/1 votes, so its pooled AUC cannot be close to 1 after a single error.
  - The small AUC difference (0.977 leaky vs 0.969 honest) is therefore partly a product of this rule.
- **Sensitivity at 0.5 is the cleaner evidence of the leak:** 0.936 vs 0.782. Use that in the report, not the AUC gap.

## 3. Moderate issues

- **Two items from the brief were only partly addressed:**
  - *Problem 2 (fold change ignores within-class variance):* elastic net and shrunken centroids sidestep it, but KNN with a t-statistic or signal-to-noise ranking was not tried.
  - *Problem 6 (panel size read off the test set):* no nested panel-size curve was run. The appendix shows only that the models, choosing for themselves, used many genes (elastic net 83; shrunken centroids 886-3051).
  - A simpler earlier check (`bs0004/golub/golub_value_add.R`) is supporting but not nested. It used repeated CV with genes re-ranked inside folds, and gave balanced accuracy of 0.53 with 1 gene, 0.86 with 3 and 0.93 with 50.
- **The shrunken-centroid result is unstable.**
  - Shrinkage was chosen on a one-patient difference in inner CV error.
  - Across the outer folds the number of genes ranged from 1 to 2953.
  - Its test posteriors all round to 0.000 or 1.000, so its misses are confidently wrong and threshold tuning has no effect.
  - Treat its numbers as the least reliable of the three models.
- **The gene-stability frequencies are optimistic.**
  - Any two 80% subsamples share most of their 31 patients, so high frequencies partly reflect overlapping data.
  - Elastic net selected about 81 probes per fit (lambda.min); 10 probes sit at the 1.00 ceiling, so they cannot be ranked against each other.
  - Elastic net keeps one probe from a group of correlated probes. A low frequency (for example TCL1A 0.08) means "a correlated probe was chosen instead", not "not informative".
- **The permutation test answers an easy question.** It confirms that there is signal, which nobody doubted for ALL vs AML. Its p-value is at the floor (1/201), so it cannot show how strong the signal is. It is still worth keeping as a check against leakage.
- **The tuned cut-offs are noisy.** Inner folds hold about 7 AML, and each Youden cut-off is estimated from about 30 probabilities. Elastic net cut-offs ranged from 0.089 to 0.683 across outer fits. glmnet warned about the small class (21 warnings in the run log).
- **The canary checks are tautological.** The functions never receive the protected rows, so the checks confirm the design rather than test it independently. RESULTS.md says so; keep that wording.

## 4. Minor points

- **Pooled AUC mixes folds.** Pooled out-of-fold AUC combines predictions from five different fold models, whose probabilities may not be on the same scale. The fold-averaged AUC (A4) is close to it, so this did not change anything here.
- **Bootstrap AUC intervals ignore training variability.** They treat the fitted model as fixed and resample only the test patients, so they do not include variability from refitting the model.
- **Gene symbol.** D88270_at is VPREB1 in GEO's GPL80 annotation, but IGL in the main report's Table 2. The probe lies in the immunoglobulin lambda region, so both names have some basis. The report should say which annotation it used.

## 5. Errors found in RESULTS.md (now corrected)

1. **"Patient 66 was missed by every model at both cut-offs, with both seeds."**
   - Wrong. Leaky KNN at its Youden cut-off caught patient 66 at seed 2026, and elastic net at its Youden cut-off caught them at seed 2028.
   - Correct statement: every honest model missed patient 66 at 0.5 with both seeds.
2. **"Patient 67 became a false positive at the tuned cut-offs" (seed 2028).**
   - Incomplete. Shrunken centroids also called patient 67 AML at 0.5.
3. **Blood errors at seed 2028: "two of those errors were patient 67".**
   - Wrong. Patient 67 accounted for four of the five blood errors; the fifth was patient 64 (AML, CCG), missed by honest KNN at 0.5.
4. **"Honest KNN caught 13/14 because it chose k = 7."**
   - Causal claim not tested. It now says this is the most likely reason.

RESULTS.md now also includes the training-set confounding, the exact intervals and the note on extreme shrunken-centroid posteriors.

## 6. What the report can and cannot claim

**Can claim:**
- Honest nested CV confirms strong ALL vs AML separation: AUC 0.969-0.997 depending on the model.
- The original CV AUC of 1.000 was inflated by ranking genes before CV.
- Learning the cut-off in training CV fixes most missed AML for KNN.
- Three different models agree on the test set (AUC 0.993-0.998), and none is better than the others.

**Cannot claim:**
- That one model is best.
- That 2-5 genes are enough. This was never chosen without the test set.
- That blood works as well as marrow. There are 10 blood samples, and hospital is mixed with sample type.
- That the model would work equally well at a new hospital. The misses all come from new hospitals, and training hospital equals diagnosis.

## 7. Suggested fixes, by effort

| Fix | Effort | Effect |
|---|---|---|
| Quote exact (Clopper-Pearson) intervals for recall, specificity and accuracy | Small (change `boot_ci`) | Honest uncertainty |
| Rename CV ranges "range across repeats" in tables and figure captions | Small | Avoids misreading |
| Add hospital to the limitations in both reports | Small | Covers the main confounder |
| Repeat the final-model inner CV (e.g. 10 x 5-fold) before choosing k, lambda, shrinkage and cut-off | Medium | Final models no longer depend on one split |
| Nested panel-size curve (1-50 genes, ranking inside folds) | Medium | Answers problem 6 properly |
| KNN with t-statistic ranking through the same nested CV | Medium | Answers problem 2 |
| Shrunken centroids: choose shrinkage with a one-standard-error rule | Small | More stable gene numbers (a post-hoc change, so it must be labelled as such) |
