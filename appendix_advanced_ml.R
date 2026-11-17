###############################################################################
#  APPENDIX: more rigorous machine learning on the Golub (1999) ALL vs AML data
#
#  This script reproduces every appendix result from a clean R session:
#     Rscript appendix_advanced_ml.R          (seeds 2026 and 2028)
#     Rscript appendix_advanced_ml.R 2026     (one seed only)
#
#  What it adds to the main project script (which is NOT modified):
#    A. Nested cross-validation on the 38 training patients
#       outer: stratified 5-fold x 20 repeats
#       inner: stratified 5-fold, REPEATED 10 times (version 2), so that the
#              chosen settings and cut-offs do not depend on one random split
#    B. Models, all run through the same nested procedure:
#         KNN as in our script (top 50 genes by |log2 FC|): honest and leaky
#         KNN with a t-statistic ranking instead of fold change     (version 2)
#         KNN with the panel size (1-50 genes) tuned inside CV      (version 2)
#         Elastic net logistic regression  (PRE-DECLARED PRIMARY MODEL)
#         Nearest shrunken centroids (pamr), lowest-error rule (pre-declared)
#         Nearest shrunken centroids, one-standard-error rule  (version 2, POST HOC)
#    C. Decision threshold: default 0.5 and Youden's J on inner-fold
#       out-of-fold probabilities
#    D. Independent test set (34 patients) used ONCE per model
#    E. Test-set uncertainty: exact Clopper-Pearson CIs for proportions
#       (version 2), stratified bootstrap CI for AUC; label-permutation test
#    F. Gene selection stability (100 stratified 80% subsamples)
#    G. Bone marrow vs peripheral blood breakdown
#    H. Seed sensitivity (whole analysis rerun with seed 2028)
#    J. Hospital: training hospital = diagnosis; test results by hospital (v2)
#    K. Nested gene-panel-size curve, 1 to 50 genes (version 2)
#
#  Kept identical to the main script: golubEsets data and split, AML = positive
#  class, floor 100 / ceiling 16000 / log10, variation filter (max/min > 5 and
#  max - min > 500), |log2 fold change| ranking for KNN, seed 2026.
#
#  Leakage rule used throughout: every fitting function receives ONLY the rows
#  of its fitting data (X[train, ], y[train]). It never sees validation or test
#  rows, so it cannot use them. Section I re-checks this with "canary" tests.
###############################################################################


# =============================================================================
# STEP 0: PACKAGES AND SETTINGS
# =============================================================================
# install.packages(c("glmnet", "pamr", "caret", "pROC")); BiocManager::install("golubEsets")
suppressPackageStartupMessages({
  library(Biobase)      # exprs() and pData()
  library(golubEsets)   # the Golub data
  library(glmnet)       # elastic net
  library(pamr)         # nearest shrunken centroids
  library(caret)        # only used to reproduce the original leaky KNN estimate
  library(pROC)         # ROC curves and AUC
  library(parallel)     # mclapply (runs on Linux / macOS)
})

N_CORES     <- max(1, min(28, detectCores() - 2))
N_REP       <- 20      # outer cross-validation repeats
N_OUTER     <- 5       # outer folds
N_INNER     <- 5       # inner folds
N_INNER_REP <- 10      # inner CV repeats (version 2)
N_BOOT      <- 2000    # bootstrap resamples on the test set (AUC only)
N_PERM      <- 200     # label permutations (brief asks for at least 100)
N_SUB       <- 100     # 80% subsamples for gene stability
KNN_GENES   <- 50      # as in our script
PANEL_SIZES <- c(1, 2, 3, 5, 10, 20, 50)   # as in our script's Figure 6
K_MAX       <- 15      # as in our script: odd k from 1 to 15
ALPHA       <- 0.5     # elastic net mixing
QUICK       <- Sys.getenv("QUICK") == "1"   # smoke test only: tiny settings, separate output folder
if (QUICK) { N_REP <- 2; N_INNER_REP <- 2; N_BOOT <- 50; N_PERM <- 4; N_SUB <- 4 }

ORIGINAL_TOP10 <- c("X82240_rna1_at", "M84526_at", "X76223_s_at", "M27891_at",
                    "M30703_s_at", "L33930_s_at", "D88270_at", "M57731_s_at",
                    "M20203_s_at", "M27783_s_at")
RANGE_LABEL <- "mean (2.5th-97.5th percentile range across 20 CV repeats; NOT a confidence interval)"


# =============================================================================
# STEP 1: DATA AND PREPROCESSING (identical to the main script)
# =============================================================================
data(Golub_Train)
data(Golub_Test)

X_train <- t(exprs(Golub_Train))      # 38 x 7129
X_test  <- t(exprs(Golub_Test))       # 34 x 7129
y_train <- factor(Golub_Train$ALL.AML, levels = c("ALL", "AML"))
y_test  <- factor(Golub_Test$ALL.AML,  levels = c("ALL", "AML"))
source_test   <- as.character(Golub_Test$BM.PB)
hospital_train <- as.character(Golub_Train$Source)
hospital_test  <- as.character(Golub_Test$Source)

stopifnot(table(y_train) == c(27, 11), table(y_test) == c(20, 14),
          all(Golub_Train$BM.PB == "BM"),
          table(y_test, source_test)["ALL", ] == c(14, 6),
          table(y_test, source_test)["AML", ] == c(10, 4))

# fixed thresholds: no information is learned from the data here
X_train[X_train < 100] <- 100;  X_train[X_train > 16000] <- 16000
X_test[X_test < 100]   <- 100;  X_test[X_test > 16000]   <- 16000
L_train <- log10(X_train)
L_test  <- log10(X_test)


# =============================================================================
# STEP 2: BUILDING BLOCKS (each one only ever sees its fitting rows)
# =============================================================================

# variation filter (Dudoit et al. 2002), computed on the rows it is given
variation_filter <- function(Xraw) {
  mx <- apply(Xraw, 2, max); mn <- apply(Xraw, 2, min)
  colnames(Xraw)[(mx / mn > 5) & (mx - mn > 500)]
}

# |log2 fold change| ranking, exactly as in the main script, on given rows
rank_fold_change <- function(Xraw, Xlog, y) {
  genes <- variation_filter(Xraw)
  fc <- abs(log2(colMeans(Xraw[y == "AML", genes, drop = FALSE]) /
                 colMeans(Xraw[y == "ALL", genes, drop = FALSE])))
  names(sort(fc, decreasing = TRUE))
}

# version 2: |Welch t-statistic| on the log10 scale (accounts for the spread
# within each class, which fold change ignores)
rank_t_stat <- function(Xraw, Xlog, y) {
  genes <- variation_filter(Xraw)
  a <- Xlog[y == "AML", genes, drop = FALSE]; b <- Xlog[y == "ALL", genes, drop = FALSE]
  t <- (colMeans(a) - colMeans(b)) /
       sqrt(apply(a, 2, var) / nrow(a) + apply(b, 2, var) / nrow(b) + 1e-12)
  names(sort(abs(t), decreasing = TRUE))
}

# stratified folds by patient; every fold must contain both classes
strat_folds <- function(y, k) {
  f <- integer(length(y))
  for (cl in levels(y)) {
    idx <- which(y == cl)
    idx <- idx[sample.int(length(idx))]          # safe even for one element
    f[idx] <- rep_len(sample.int(k), length(idx))
  }
  folds <- split(seq_along(y), f)
  for (v in folds) stopifnot(length(unique(y[v])) == 2,           # validation fold
                             length(unique(y[-v])) == 2)          # fitting part
  unname(folds)
}
# version 2: several independent sets of inner folds
inner_sets <- function(y) lapply(seq_len(N_INNER_REP), function(i) strat_folds(y, N_INNER))

# Youden's J threshold from out-of-fold probabilities.
# Candidate cut-offs are midpoints between observed values; a patient is called
# AML when p >= cut. Ties in J are broken by the cut closest to 0.5 (declared
# in advance, so that the default is kept unless the data argue against it).
youden_cut <- function(y, p) {
  u <- sort(unique(p))
  cand <- if (length(u) == 1) u else c(u[1] - 1e-6, (u[-1] + u[-length(u)]) / 2, u[length(u)] + 1e-6)
  J <- sapply(cand, function(cc) mean(p[y == "AML"] >= cc) + mean(p[y == "ALL"] < cc) - 1)
  best <- cand[abs(J - max(J)) < 1e-12]
  best[which.min(abs(best - 0.5))]
}

# kNN vote probability: share of AML among the k nearest fitting patients
# (Euclidean distance on z-scores; z-score mean and SD from fitting rows only)
knn_votes <- function(Ztr, ytr, Znew, ks) {
  d <- sqrt(pmax(outer(rowSums(Znew^2), rowSums(Ztr^2), "+") - 2 * Znew %*% t(Ztr), 0))
  ord <- t(apply(d, 1, order))
  if (nrow(Znew) == 1) ord <- matrix(ord, nrow = 1)
  sapply(ks, function(k) rowMeans(matrix(ytr[ord[, 1:k, drop = FALSE]] == "AML", nrow = nrow(Znew))))
}
zfit   <- function(X) list(mu = colMeans(X), s = pmax(apply(X, 2, sd), 1e-8))
zapply <- function(X, z) sweep(sweep(X, 2, z$mu, "-"), 2, z$s, "/")
auc_of <- function(y, p) as.numeric(auc(y, p, levels = c("ALL", "AML"), direction = "<", quiet = TRUE))


# -----------------------------------------------------------------------------
# MODEL 1: KNN
#   genes = NULL  -> honest: ranking redone inside the fitting rows and again
#                    inside every inner fold
#   genes = <vec> -> leaky reproduction: genes fixed in advance from all 38
#   rank_fun: rank_fold_change (as in our script) or rank_t_stat (version 2)
#   sizes: number of top genes; one value = fixed panel, several = tuned
#   k grid: odd k <= 15 AND k < 2 x (AML patients in the smallest inner fitting
#   set), so that AML can always win a majority vote.
#   Choice: highest inner out-of-fold AUC, averaged over the inner repeats;
#   ties -> lowest Brier score -> fewer genes -> smallest k (declared in advance).
# -----------------------------------------------------------------------------
fit_knn <- function(Xr, Xl, y, inner, genes = NULL, rank_fun = rank_fold_change, sizes = KNN_GENES) {
  min_aml <- min(sapply(unlist(inner, recursive = FALSE), function(v) sum(y[-v] == "AML")))
  ks <- seq(1, K_MAX, by = 2); ks <- ks[ks < 2 * min_aml]
  if (!is.null(genes)) sizes <- length(genes)
  oof <- array(NA, c(length(inner), length(y), length(sizes), length(ks)))
  for (r in seq_along(inner)) for (v in inner[[r]]) {
    rk <- if (is.null(genes)) rank_fun(Xr[-v, , drop = FALSE], Xl[-v, , drop = FALSE], y[-v]) else genes
    for (si in seq_along(sizes)) {
      g <- rk[1:sizes[si]]
      z <- zfit(Xl[-v, g, drop = FALSE])
      oof[r, v, si, ] <- knn_votes(zapply(Xl[-v, g, drop = FALSE], z), y[-v], zapply(Xl[v, g, drop = FALSE], z), ks)
    }
  }
  aucs  <- apply(oof, c(1, 3, 4), function(p) auc_of(y, p))           # repeat x size x k
  brier <- apply(oof, c(1, 3, 4), function(p) mean((p - (y == "AML"))^2))
  A <- apply(aucs, c(2, 3), mean); B <- apply(brier, c(2, 3), mean)
  grid <- expand.grid(si = seq_along(sizes), ki = seq_along(ks))
  grid$auc <- A[cbind(grid$si, grid$ki)]; grid$brier <- B[cbind(grid$si, grid$ki)]
  cand <- grid[abs(grid$auc - max(grid$auc)) < 1e-12, ]
  best <- cand[order(cand$brier, sizes[cand$si], ks[cand$ki]), ][1, ]
  p_oof <- colMeans(matrix(oof[, , best$si, best$ki], nrow = length(inner)))   # averaged over inner repeats
  rk <- if (is.null(genes)) rank_fun(Xr, Xl, y) else genes
  g  <- rk[1:sizes[best$si]]
  z  <- zfit(Xl[, g, drop = FALSE])
  list(model = "KNN", genes = g, z = z, Ztr = zapply(Xl[, g, drop = FALSE], z), ytr = y,
       k = ks[best$ki], n_k_tied = nrow(cand), inner_auc = best$auc,
       cut = youden_cut(y, p_oof))
}
predict_knn <- function(fit, Xl) as.vector(knn_votes(fit$Ztr, fit$ytr, zapply(Xl[, fit$genes, drop = FALSE], fit$z), fit$k))


# -----------------------------------------------------------------------------
# MODEL 2: ELASTIC NET logistic regression (primary model)
#   genes: variation filter on the fitting rows; glmnet standardises each gene
#   with fitting-row statistics. alpha = 0.5. Class weights = n / (2 n_class).
#   lambda: binomial deviance averaged over the repeated inner CV; lambda.min.
#   Threshold: Youden on the inner out-of-fold probabilities at that lambda,
#   averaged over the inner repeats.
#   Note: inner folds share the filtered gene list and the lambda sequence of
#   their fitting rows; the filter uses no labels.
# -----------------------------------------------------------------------------
fit_enet <- function(Xr, Xl, y, inner) {
  g <- variation_filter(Xr); x <- Xl[, g, drop = FALSE]
  w <- ifelse(y == "AML", length(y) / (2 * sum(y == "AML")), length(y) / (2 * sum(y == "ALL")))
  full <- glmnet(x, y, family = "binomial", alpha = ALPHA, weights = w, standardize = TRUE)
  lam <- full$lambda
  cvm <- matrix(NA, length(inner), length(lam)); pre <- array(NA, c(length(inner), length(y), length(lam)))
  for (r in seq_along(inner)) {
    foldid <- integer(length(y)); for (i in seq_along(inner[[r]])) foldid[inner[[r]][[i]]] <- i
    cv <- cv.glmnet(x, y, family = "binomial", alpha = ALPHA, weights = w, lambda = lam,
                    foldid = foldid, type.measure = "deviance", keep = TRUE, standardize = TRUE)
    j <- match(cv$lambda, lam)
    cvm[r, j] <- cv$cvm
    # glmnet stores binomial out-of-fold fits as log-odds (checked against a
    # manual fold refit), so convert them to probabilities
    pre[r, , j] <- plogis(cv$fit.preval[, seq_along(j)])
  }
  li  <- which.min(colMeans(cvm))                  # lambdas missing in any repeat are skipped (NA)
  oof <- colMeans(matrix(pre[, , li], nrow = length(inner)))
  stopifnot(!anyNA(oof))
  b <- as.matrix(coef(full, s = lam[li], exact = FALSE))[-1, 1]
  list(model = "Elastic net", genes = g, fit = full, lambda = lam[li],
       selected = names(b)[b != 0], coef = b[b != 0], cut = youden_cut(y, oof))
}
predict_enet <- function(fit, Xl) as.vector(predict(fit$fit, Xl[, fit$genes, drop = FALSE], s = fit$lambda, type = "response"))


# -----------------------------------------------------------------------------
# MODEL 3: NEAREST SHRUNKEN CENTROIDS (pamr)
#   genes: variation filter on the fitting rows; pamr standardises internally.
#   Equal class priors (0.5, 0.5). Inner CV error averaged over inner repeats.
#   rule = "min": lowest error, ties -> larger threshold (PRE-DECLARED)
#   rule = "1se": largest threshold with error <= min + SE, SE = sqrt(e(1-e)/n)
#                 (version 2, POST HOC: added after seeing the min rule pick
#                 no shrinkage on a one-patient difference)
#   Cut-off: Youden on inner out-of-fold posteriors at the chosen threshold.
# -----------------------------------------------------------------------------
quiet <- function(expr) { out <- NULL; invisible(capture.output(out <- expr)); out }
fit_nsc <- function(Xr, Xl, y, inner, rule = "min") {
  g <- variation_filter(Xr)
  dat <- list(x = t(Xl[, g, drop = FALSE]), y = y, geneid = g, genenames = g)
  fit <- quiet(pamr.train(dat, prior = c(0.5, 0.5)))
  cvs <- lapply(inner, function(fs) quiet(pamr.cv(fit, dat, folds = fs)))
  err <- colMeans(do.call(rbind, lapply(cvs, `[[`, "error")))
  emin <- min(err)
  ti <- if (rule == "min") max(which(abs(err - emin) < 1e-12)) else
          max(which(err <= emin + sqrt(emin * (1 - emin) / length(y)) + 1e-12))
  thr <- fit$threshold[ti]
  oof <- rowMeans(sapply(cvs, function(cv) cv$prob[, 2, ti]))   # column 2 = AML
  nz  <- quiet(pamr.predict(fit, dat$x, threshold = thr, type = "nonzero"))
  list(model = "Shrunken centroids", genes = g, fit = fit, threshold = thr,
       selected = g[nz], cut = youden_cut(y, oof))
}
predict_nsc <- function(fit, Xl) as.vector(quiet(pamr.predict(fit$fit, t(Xl[, fit$genes, drop = FALSE]),
                                                              threshold = fit$threshold, type = "posterior"))[, 2])


# one entry point for all procedures; receives ONLY fitting rows
fit_model <- function(m, Xr, Xl, y, inner, leaky_genes = NULL) {
  if (startsWith(m, "knn_size_"))                     # K: fixed panel size, |log2 FC| ranking
    return(fit_knn(Xr, Xl, y, inner, sizes = as.integer(sub("knn_size_", "", m))))
  switch(m,
    knn_leaky  = fit_knn(Xr, Xl, y, inner, genes = leaky_genes),
    knn_honest = fit_knn(Xr, Xl, y, inner),
    knn_t      = fit_knn(Xr, Xl, y, inner, rank_fun = rank_t_stat),
    knn_panel  = fit_knn(Xr, Xl, y, inner, sizes = PANEL_SIZES),
    enet       = fit_enet(Xr, Xl, y, inner),
    nsc        = fit_nsc(Xr, Xl, y, inner, rule = "min"),
    nsc_1se    = fit_nsc(Xr, Xl, y, inner, rule = "1se"),
    stop("unknown model ", m))
}
predict_model <- function(m, fit, Xl) {
  if (startsWith(m, "knn")) return(predict_knn(fit, Xl))
  switch(m, enet = predict_enet(fit, Xl), nsc = , nsc_1se = predict_nsc(fit, Xl))
}

MODELS <- c(knn_leaky  = "KNN, leaky (genes ranked on all 38)",
            knn_honest = "KNN, honest, |log2 FC| (50 genes)",
            knn_t      = "KNN, honest, t-statistic (50 genes)",
            knn_panel  = "KNN, honest, panel size tuned (1-50 genes)",
            enet       = "Elastic net (primary)",
            nsc        = "Shrunken centroids, min-error rule",
            nsc_1se    = "Shrunken centroids, 1-SE rule (post hoc)")
SIZE_MODELS <- setNames(paste0("knn_size_", PANEL_SIZES), paste(PANEL_SIZES, "genes"))

# performance at a cut-off (AML = positive); cut may differ per patient
perf <- function(y, p, cut) {
  pred <- p >= cut
  TP <- sum(pred & y == "AML"); FN <- sum(!pred & y == "AML")
  TN <- sum(!pred & y == "ALL"); FP <- sum(pred & y == "ALL")
  a  <- if (length(unique(y)) == 2) auc_of(y, p) else NA
  c(AUC = a, sensitivity = TP / (TP + FN), specificity = TN / (TN + FP),
    accuracy = (TP + TN) / length(y), balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2,
    TP = TP, FN = FN, TN = TN, FP = FP)
}
# exact (Clopper-Pearson) 95% CI for a proportion, as text
cp <- function(x, n) { if (n == 0) return("n/a"); ci <- binom.test(x, n)$conf.int
  sprintf("%.3f (%.3f-%.3f)", x / n, ci[1], ci[2]) }
cp_lo <- function(x, n) binom.test(x, n)$conf.int[1]
cp_hi <- function(x, n) binom.test(x, n)$conf.int[2]


# =============================================================================
# STEP 3: NESTED CROSS-VALIDATION (one call = one model x one set of labels)
# =============================================================================
make_plan <- function(y) {
  # all random fold assignments are drawn here, in the main process, so the
  # results do not depend on the number of cores
  lapply(1:N_REP, function(r) {
    outer <- strat_folds(y, N_OUTER)
    list(outer = outer, inner = lapply(outer, function(v) inner_sets(y[-v])))
  })
}

nested_cv <- function(m, y, plan, leaky_genes = NULL, cores = N_CORES) {
  jobs <- expand.grid(f = 1:N_OUTER, r = 1:N_REP)
  res <- mclapply(seq_len(nrow(jobs)), function(j) {
    r <- jobs$r[j]; f <- jobs$f[j]
    v  <- plan[[r]]$outer[[f]]; tr <- setdiff(seq_along(y), v)
    fit <- fit_model(m, X_train[tr, , drop = FALSE], L_train[tr, , drop = FALSE], y[tr],
                     plan[[r]]$inner[[f]], leaky_genes)
    data.frame(rep = r, fold = f, patient = v, y = y[v],
               p = predict_model(m, fit, L_train[v, , drop = FALSE]), cut = fit$cut,
               k = if (!is.null(fit$k)) fit$k else NA,
               n_k_tied = if (!is.null(fit$n_k_tied)) fit$n_k_tied else NA,
               n_genes = if (!is.null(fit$selected)) length(fit$selected) else length(fit$genes))
  }, mc.cores = cores, mc.preschedule = FALSE)
  bad <- sapply(res, inherits, "try-error"); if (any(bad)) stop(res[[which(bad)[1]]])
  oof <- do.call(rbind, res)
  stopifnot(nrow(oof) == N_REP * length(y))              # every patient once per repeat
  by_rep <- do.call(rbind, lapply(split(oof, oof$rep), function(d)
    rbind(data.frame(rep = d$rep[1], threshold = "0.5", t(perf(d$y, d$p, 0.5))),
          data.frame(rep = d$rep[1], threshold = "Youden (inner CV)", t(perf(d$y, d$p, d$cut))))))
  list(oof = oof, by_rep = by_rep)
}

# summary across repeats: mean and percentile RANGE (not a confidence interval)
summarise_cv <- function(cv, label) {
  d <- cv$by_rep
  do.call(rbind, lapply(split(d, d$threshold), function(s) {
    st <- function(x) sprintf("%.3f (%.3f-%.3f)", mean(x), quantile(x, 0.025), quantile(x, 0.975))
    data.frame(model = label, threshold = s$threshold[1],
               AUC = st(s$AUC), sensitivity = st(s$sensitivity), specificity = st(s$specificity),
               accuracy = st(s$accuracy), balanced_accuracy = st(s$balanced_accuracy),
               AUC_mean = mean(s$AUC), AUC_sd = sd(s$AUC), balacc_mean = mean(s$balanced_accuracy),
               balacc_lo = quantile(s$balanced_accuracy, 0.025), balacc_hi = quantile(s$balanced_accuracy, 0.975),
               AUC_lo = quantile(s$AUC, 0.025), AUC_hi = quantile(s$AUC, 0.975),
               sens_mean = mean(s$sensitivity))
  }))
}


# =============================================================================
# STEP 4: THE WHOLE ANALYSIS FOR ONE SEED
# =============================================================================
run_analysis <- function(seed, outdir, do_figures = TRUE) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  set.seed(seed)
  log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
  out <- list(seed = seed)
  readme <- c("Column note for the A2 and K tables:",
              paste("Every CV value is the", RANGE_LABEL, "."),
              "Test-set proportions (recall, specificity, accuracy) use exact Clopper-Pearson 95% CIs;",
              "test-set AUC uses a stratified bootstrap 95% CI (2000 resamples).")
  writeLines(readme, file.path(outdir, "README_columns.txt"))

  # ---- A1. the original leaky estimate, reproduced exactly with caret -------
  # genes ranked once on all 38 patients, z-scored on all 38, then 5-fold CV
  ranked_all <- rank_fold_change(X_train, L_train, y_train)
  stopifnot(identical(ranked_all[1:10], ORIGINAL_TOP10))   # same genes as our report
  g50 <- ranked_all[1:KNN_GENES]
  Z50 <- scale(L_train[, g50])
  set.seed(seed)
  caret_fit <- train(x = as.data.frame(Z50), y = y_train, method = "knn",
                     trControl = trainControl(method = "cv", number = 5, classProbs = TRUE,
                                              summaryFunction = twoClassSummary),
                     tuneGrid = expand.grid(k = seq(1, 15, by = 2)), metric = "ROC")
  out$caret_original <- data.frame(k = caret_fit$results$k, cv_AUC = round(caret_fit$results$ROC, 3),
                                   chosen = caret_fit$results$k == caret_fit$bestTune$k)
  write.csv(out$caret_original, file.path(outdir, "A1_original_caret_leaky_cv.csv"), row.names = FALSE)
  log_msg("original caret CV AUC by k:", paste(out$caret_original$cv_AUC, collapse = " "), "| chosen k =", caret_fit$bestTune$k)

  # ---- A2. nested CV for every procedure (same folds for every model) -------
  set.seed(seed)
  plan <- make_plan(y_train)
  out$cv <- list()
  for (m in c(names(MODELS), SIZE_MODELS)) {
    t0 <- Sys.time()
    out$cv[[m]] <- nested_cv(m, y_train, plan, leaky_genes = if (m == "knn_leaky") g50 else NULL)
    log_msg("nested CV done:", m, round(difftime(Sys.time(), t0, units = "mins"), 1), "min")
  }
  cv_summary <- do.call(rbind, lapply(names(MODELS), function(m) summarise_cv(out$cv[[m]], MODELS[[m]])))
  rownames(cv_summary) <- NULL
  out$cv_summary <- cv_summary
  names_out <- names(cv_summary); names_out[3:7] <- paste0(names_out[3:7], " [", RANGE_LABEL, "]")
  write.csv(setNames(cv_summary[, 1:7], names_out[1:7]), file.path(outdir, "A2_nested_cv_performance.csv"), row.names = FALSE)
  write.csv(do.call(rbind, lapply(names(MODELS), function(m) cbind(model = MODELS[[m]], out$cv[[m]]$by_rep))),
            file.path(outdir, "A2_nested_cv_by_repeat.csv"), row.names = FALSE)
  # caret (and so our original script) reports the AUC averaged over the 5
  # validation folds of about 8 patients each; the table above pools every
  # out-of-fold prediction first. Show both, so the two effects can be told apart.
  fold_avg <- do.call(rbind, lapply(names(MODELS), function(m) {
    o <- out$cv[[m]]$oof
    fa <- sapply(split(o, list(o$rep, o$fold)), function(d) auc_of(d$y, d$p))
    pooled <- out$cv[[m]]$by_rep$AUC[out$cv[[m]]$by_rep$threshold == "0.5"]
    data.frame(model = MODELS[[m]], pooled_oof_AUC_mean = round(mean(pooled), 3),
               fold_averaged_AUC_mean = round(mean(fa), 3),
               share_of_folds_with_AUC_1 = round(mean(fa == 1), 3))
  }))
  out$fold_avg <- fold_avg
  write.csv(fold_avg, file.path(outdir, "A4_pooled_vs_fold_averaged_auc.csv"), row.names = FALSE)

  # tuning behaviour inside the outer folds
  tun <- do.call(rbind, lapply(names(MODELS), function(m) {
    o <- unique(out$cv[[m]]$oof[, c("rep", "fold", "cut", "k", "n_k_tied", "n_genes")])
    tabstr <- function(x) paste(names(table(x)), table(x), sep = ":", collapse = " ")
    data.frame(model = MODELS[[m]], fits = nrow(o),
               k_values = if (all(is.na(o$k))) "n/a" else tabstr(o$k),
               fits_with_tied_choice = if (all(is.na(o$n_k_tied))) NA else sum(o$n_k_tied > 1),
               genes_values = if (m == "knn_panel") tabstr(o$n_genes) else "",
               genes_median = median(o$n_genes), genes_range = paste(range(o$n_genes), collapse = "-"),
               youden_cut_median = round(median(o$cut), 3), youden_cut_range = paste(round(range(o$cut), 3), collapse = " to "))
  }))
  out$tuning <- tun
  write.csv(tun, file.path(outdir, "A3_nested_cv_tuning.csv"), row.names = FALSE)

  # ---- D. refit on all 38, predict the test set ONCE per model ---------------
  set.seed(seed)
  inner_all <- inner_sets(y_train)
  out$final <- list(); out$test_p <- list()
  for (m in c(names(MODELS), SIZE_MODELS)) {
    out$final[[m]]  <- fit_model(m, X_train, L_train, y_train, inner_all, leaky_genes = g50)
    out$test_p[[m]] <- predict_model(m, out$final[[m]], L_test)
  }
  final_desc <- do.call(rbind, lapply(names(MODELS), function(m) {
    f <- out$final[[m]]
    hp <- if (startsWith(m, "knn")) sprintf("k = %d, %d genes (%d settings tied on inner AUC)", f$k, length(f$genes), f$n_k_tied) else
          if (m == "enet") sprintf("lambda = %.4g", f$lambda) else sprintf("shrinkage threshold = %.3f", f$threshold)
    data.frame(model = MODELS[[m]], hyperparameter = hp,
               genes_used = if (!is.null(f$selected)) length(f$selected) else length(f$genes),
               youden_cut = round(f$cut, 3))
  }))
  out$final_desc <- final_desc
  write.csv(final_desc, file.path(outdir, "D1_final_models.csv"), row.names = FALSE)

  # ---- E1. test-set uncertainty ------------------------------------------------
  # proportions: exact Clopper-Pearson CI (version 2; the bootstrap was too
  # narrow with 14 AML); AUC: stratified bootstrap CI
  set.seed(seed)
  ia <- which(y_test == "AML"); il <- which(y_test == "ALL")
  boot_idx <- replicate(N_BOOT, c(il[sample.int(length(il), replace = TRUE)], ia[sample.int(length(ia), replace = TRUE)]))
  boot_auc <- function(p, idx_set = boot_idx, yy = y_test) {
    B <- apply(idx_set, 2, function(b) auc_of(yy[b], p[b]))
    quantile(B, c(0.025, 0.975), na.rm = TRUE)
  }
  test_row <- function(label, th, cut, p, yy = y_test, idx_set = boot_idx) {
    e <- perf(yy, p, cut); nA <- sum(yy == "AML"); nL <- sum(yy == "ALL")
    ci <- if (nA > 0 && nL > 0) boot_auc(p, idx_set, yy) else c(NA, NA)
    data.frame(model = label, threshold = th, cut_off = round(cut, 3),
      AUC_bootstrap_CI = if (is.na(e["AUC"])) "n/a" else sprintf("%.3f (%.3f-%.3f)", e["AUC"], ci[1], min(1, ci[2])),
      AML_recall_exact_CI = cp(e["TP"], nA), specificity_exact_CI = cp(e["TN"], nL),
      accuracy_exact_CI = cp(e["TP"] + e["TN"], length(yy)),
      TP = e["TP"], FN = e["FN"], TN = e["TN"], FP = e["FP"],
      sens_est = e["sensitivity"], sens_lo = cp_lo(e["TP"], nA), sens_hi = cp_hi(e["TP"], nA),
      AUC_est = e["AUC"], AUC_lo = ci[1], AUC_hi = ci[2])
  }
  ths <- c("0.5", "Youden (training inner CV)")
  out$test <- do.call(rbind, lapply(names(MODELS), function(m) do.call(rbind, lapply(ths, function(th)
    test_row(MODELS[[m]], th, if (th == "0.5") 0.5 else out$final[[m]]$cut, out$test_p[[m]])))))
  rownames(out$test) <- NULL
  write.csv(out$test[, 1:7], file.path(outdir, "D2_test_performance_ci.csv"), row.names = FALSE)
  write.csv(out$test[, c(1:3, 8:11)], file.path(outdir, "D3_test_confusion_matrices.csv"), row.names = FALSE)
  # per-patient test probabilities (for transparency)
  write.csv(data.frame(patient = rownames(L_test), class = y_test, source = source_test, hospital = hospital_test,
                       sapply(names(MODELS), function(m) round(out$test_p[[m]], 3))),
            file.path(outdir, "D4_test_probabilities.csv"), row.names = FALSE)
  # which test patients were misclassified, and where did they come from?
  wrong <- do.call(rbind, lapply(names(MODELS), function(m) do.call(rbind, lapply(ths, function(th) {
    cut <- if (th == "0.5") 0.5 else out$final[[m]]$cut
    i <- which((out$test_p[[m]] >= cut) != (y_test == "AML"))
    if (!length(i)) return(NULL)
    data.frame(model = MODELS[[m]], threshold = th, patient = rownames(L_test)[i], class = y_test[i],
               source = source_test[i], hospital = hospital_test[i],
               hospital_in_training = hospital_test[i] %in% unique(hospital_train),
               p_AML = round(out$test_p[[m]][i], 3))
  }))))
  out$wrong <- wrong
  write.csv(wrong, file.path(outdir, "D5_misclassified_test_patients.csv"), row.names = FALSE)
  log_msg("test set evaluated")

  # ---- G. bone marrow vs peripheral blood ------------------------------------
  out$source <- do.call(rbind, lapply(names(MODELS), function(m) do.call(rbind, lapply(c("BM", "PB"), function(src)
    do.call(rbind, lapply(ths, function(th) {
      idx <- which(source_test == src); ys <- y_test[idx]
      a <- which(ys == "AML"); l <- which(ys == "ALL")
      bi <- replicate(N_BOOT, c(l[sample.int(length(l), replace = TRUE)], a[sample.int(length(a), replace = TRUE)]))
      r <- test_row(MODELS[[m]], th, if (th == "0.5") 0.5 else out$final[[m]]$cut, out$test_p[[m]][idx], ys, bi)
      cbind(r[, 1:2], source = src, n = length(idx), n_ALL = length(l), n_AML = length(a),
            errors = r$FN + r$FP, r[, 4:7])
    }))))))
  rownames(out$source) <- NULL
  write.csv(out$source, file.path(outdir, "G_bone_marrow_vs_blood.csv"), row.names = FALSE)

  # ---- J. hospital (version 2) -------------------------------------------------
  # training: hospital and diagnosis are the same thing (DFCI = ALL, CALGB = AML)
  tr_tab <- as.data.frame.matrix(table(hospital = hospital_train, class = y_train))
  te_tab <- as.data.frame(table(hospital = hospital_test, class = y_test, source = source_test))
  te_tab <- te_tab[te_tab$Freq > 0, ]
  write.csv(cbind(hospital = rownames(tr_tab), tr_tab), file.path(outdir, "J1_training_hospital_by_class.csv"), row.names = FALSE)
  write.csv(te_tab, file.path(outdir, "J2_test_hospital_by_class_and_source.csv"), row.names = FALSE)
  new_hosp <- !(hospital_test %in% unique(hospital_train))
  out$hospital <- do.call(rbind, lapply(names(MODELS), function(m) {
    p <- out$test_p[[m]]; aml <- y_test == "AML"
    wt <- suppressWarnings(wilcox.test(p[aml & !new_hosp], p[aml & new_hosp]))
    err <- (p >= 0.5) != aml
    data.frame(model = MODELS[[m]],
               AML_at_training_hospital_median_p = round(median(p[aml & !new_hosp]), 3),
               AML_at_new_hospitals_median_p = round(median(p[aml & new_hosp]), 3),
               wilcoxon_p_AML_probabilities = signif(wt$p.value, 2),
               StJude_ALL_max_p = round(max(p[!aml & hospital_test == "St-Jude"]), 3),
               errors_at_0.5_training_hospitals = sprintf("%d/%d", sum(err[!new_hosp]), sum(!new_hosp)),
               errors_at_0.5_new_hospitals = sprintf("%d/%d", sum(err[new_hosp]), sum(new_hosp)),
               fisher_p_errors = if (length(unique(err)) < 2) NA else signif(fisher.test(table(new_hosp, err))$p.value, 2))
  }))
  write.csv(out$hospital, file.path(outdir, "J3_test_results_by_hospital.csv"), row.names = FALSE)

  # ---- K. nested gene-panel-size curve (version 2) ---------------------------
  # each panel size is a fixed procedure (|log2 FC| ranking inside folds, k
  # tuned in inner CV); the test results per size are descriptive only: nothing
  # is chosen from them
  out$panel <- do.call(rbind, lapply(names(SIZE_MODELS), function(nm) {
    m <- SIZE_MODELS[[nm]]; s <- summarise_cv(out$cv[[m]], nm)
    s5 <- s[s$threshold == "0.5", ]; sy <- s[s$threshold != "0.5", ]
    t5 <- test_row(nm, "0.5", 0.5, out$test_p[[m]])
    data.frame(genes = as.integer(sub(" genes", "", nm)),
               cv_AUC = s5$AUC, cv_balanced_accuracy_0.5 = s5$balanced_accuracy,
               cv_balanced_accuracy_youden = sy$balanced_accuracy,
               test_AUC = t5$AUC_bootstrap_CI, test_accuracy_0.5 = t5$accuracy_exact_CI,
               test_AML_recall_0.5 = t5$AML_recall_exact_CI,
               cv_AUC_mean = s5$AUC_mean, cv_AUC_lo = s5$AUC_lo, cv_AUC_hi = s5$AUC_hi,
               cv_balacc_mean = s5$balacc_mean, cv_balacc_lo = s5$balacc_lo, cv_balacc_hi = s5$balacc_hi,
               cv_balacc_y_mean = sy$balacc_mean,
               test_acc_est = (t5$TP + t5$TN) / 34, test_AUC_est = t5$AUC_est)
  }))
  pn <- names(out$panel); pn[2:4] <- paste0(pn[2:4], " [", RANGE_LABEL, "]")
  write.csv(setNames(out$panel[, 1:7], pn[1:7]), file.path(outdir, "K_panel_size_curve.csv"), row.names = FALSE)
  log_msg("hospital and panel-size tables done")

  # ---- E2. label-permutation test for the primary model ----------------------
  set.seed(seed)
  perm_plans <- lapply(1:N_PERM, function(i) { yp <- sample(y_train); list(y = yp, plan = make_plan(yp)) })
  t0 <- Sys.time()
  null_auc <- unlist(mclapply(perm_plans, function(pp) {
    cv <- nested_cv("enet", pp$y, pp$plan, cores = 1)
    mean(cv$by_rep$AUC[cv$by_rep$threshold == "0.5"])
  }, mc.cores = N_CORES, mc.preschedule = FALSE))
  obs_auc <- mean(out$cv$enet$by_rep$AUC[out$cv$enet$by_rep$threshold == "0.5"])
  out$perm <- list(null = null_auc, observed = obs_auc,
                   p = (1 + sum(null_auc >= obs_auc)) / (1 + length(null_auc)))
  write.csv(data.frame(permutation = seq_along(null_auc), null_mean_cv_AUC = null_auc),
            file.path(outdir, "E2_permutation_null.csv"), row.names = FALSE)
  write.csv(data.frame(model = "Elastic net", n_permutations = length(null_auc), observed_mean_cv_AUC = obs_auc,
                       null_mean = mean(null_auc), null_sd = sd(null_auc), null_max = max(null_auc),
                       null_95th = quantile(null_auc, 0.95), empirical_p = out$perm$p),
            file.path(outdir, "E2_permutation_summary.csv"), row.names = FALSE)
  log_msg("permutation test done:", round(difftime(Sys.time(), t0, units = "mins"), 1), "min")

  # ---- F. gene selection stability -------------------------------------------
  set.seed(seed)
  subs <- lapply(1:N_SUB, function(i) {
    s <- unlist(lapply(levels(y_train), function(cl) { idx <- which(y_train == cl)
      idx[sample.int(length(idx), round(0.8 * length(idx)))] }))
    list(s = s, inner = inner_sets(y_train[s]))
  })
  stab_models <- c("enet", "nsc", "nsc_1se")
  sel <- mclapply(subs, function(ss) setNames(lapply(stab_models, function(m)
    fit_model(m, X_train[ss$s, ], L_train[ss$s, ], y_train[ss$s], ss$inner)$selected), stab_models),
    mc.cores = N_CORES, mc.preschedule = FALSE)
  annot <- read.delim(gzfile("annotation/GPL80.annot.gz"), skip = grep("^ID\t", readLines(gzfile("annotation/GPL80.annot.gz"), n = 200)) - 1,
                      quote = "", check.names = FALSE, stringsAsFactors = FALSE, fill = TRUE)
  sym <- setNames(annot[["Gene symbol"]], annot$ID)
  freq_tab <- do.call(rbind, lapply(stab_models, function(m) {
    f <- sort(table(unlist(lapply(sel, `[[`, m))), decreasing = TRUE) / N_SUB
    pr <- names(f)
    dirn <- ifelse(colMeans(L_train[y_train == "AML", pr, drop = FALSE]) > colMeans(L_train[y_train == "ALL", pr, drop = FALSE]), "AML", "ALL")
    d <- data.frame(model = MODELS[[m]], probe = pr, gene_symbol = ifelse(is.na(sym[pr]) | sym[pr] == "", "(none)", sym[pr]),
               selection_frequency = as.numeric(f), higher_in = dirn,
               in_original_top10 = pr %in% ORIGINAL_TOP10,
               fold_change_rank_all38 = match(pr, ranked_all),
               median_genes_selected_per_fit = median(sapply(sel, function(s) length(s[[m]]))))
    d[order(-d$selection_frequency, d$fold_change_rank_all38), ]
  }))
  rownames(freq_tab) <- NULL
  out$freq <- freq_tab
  write.csv(freq_tab, file.path(outdir, "F_gene_selection_frequency.csv"), row.names = FALSE)
  top10_tab <- data.frame(probe = ORIGINAL_TOP10, gene_symbol = sym[ORIGINAL_TOP10],
    sapply(stab_models, function(m) sapply(ORIGINAL_TOP10, function(p) mean(sapply(sel, function(s) p %in% s[[m]])))))
  names(top10_tab)[3:5] <- paste0(stab_models, "_frequency")
  out$top10 <- top10_tab
  write.csv(top10_tab, file.path(outdir, "F_original_top10_selection.csv"), row.names = FALSE)
  log_msg("gene stability done")

  # ---- I. leakage canary checks ----------------------------------------------
  # Corrupt every row the fit must not see (outer validation rows / test rows):
  # replace expression with random noise and flip the labels, refit with the
  # same call, and check that every fitted quantity is identical.
  canary <- list()
  r <- 1; f <- 1; v <- plan[[r]]$outer[[f]]; tr <- setdiff(1:38, v)
  Xr_bad <- X_train; Xl_bad <- L_train; y_bad <- y_train
  Xr_bad[v, ] <- 10^runif(length(v) * ncol(X_train), 2, 4.2); Xl_bad[v, ] <- log10(Xr_bad[v, ])
  y_bad[v] <- ifelse(y_train[v] == "AML", "ALL", "AML")
  same_fit <- function(a, b) identical(a$genes, b$genes) && identical(a$cut, b$cut) &&
    identical(a$k, b$k) && identical(a$lambda, b$lambda) && identical(a$threshold, b$threshold) &&
    identical(a$selected, b$selected)
  for (m in names(MODELS)) {
    a <- fit_model(m, X_train[tr, ], L_train[tr, ], y_train[tr], plan[[r]]$inner[[f]], g50)
    b <- fit_model(m, Xr_bad[tr, ], Xl_bad[tr, ], y_bad[tr], plan[[r]]$inner[[f]], g50)
    canary[[length(canary) + 1]] <- data.frame(check = "outer validation rows corrupted", model = MODELS[[m]],
      identical_fit = same_fit(a, b),
      identical_predictions_on_training_rows = identical(predict_model(m, a, L_train[tr, ]), predict_model(m, b, L_train[tr, ])))
    # final model: the test set is never passed to fit_model at all
    p1 <- predict_model(m, out$final[[m]], L_train); f2 <- fit_model(m, X_train, L_train, y_train, inner_all, g50)
    canary[[length(canary) + 1]] <- data.frame(check = "final model refitted (test set never passed in)", model = MODELS[[m]],
      identical_fit = same_fit(out$final[[m]], f2),
      identical_predictions_on_training_rows = identical(p1, predict_model(m, f2, L_train)))
  }
  out$canary <- do.call(rbind, canary)
  write.csv(out$canary, file.path(outdir, "I_leakage_canary_checks.csv"), row.names = FALSE)
  stopifnot(all(out$canary$identical_fit), all(out$canary$identical_predictions_on_training_rows))
  log_msg("leakage canary checks passed")

  if (do_figures) make_figures(out, "figures")
  saveRDS(out[setdiff(names(out), c("final"))], file.path(outdir, "all_results.rds"))
  out
}


# =============================================================================
# STEP 5: FIGURES (base R, as in the main script)
# =============================================================================
make_figures <- function(out, figdir) {
  dir.create(figdir, showWarnings = FALSE)
  cols <- c(knn_leaky = "grey60", knn_honest = "steelblue", knn_t = "skyblue3", knn_panel = "slateblue",
            enet = "firebrick", nsc = "darkorange", nsc_1se = "goldenrod")
  short <- c(knn_leaky = "KNN leaky", knn_honest = "KNN |log2 FC|", knn_t = "KNN t-stat", knn_panel = "KNN panel\ntuned",
             enet = "Elastic net", nsc = "Centroids\nmin-error", nsc_1se = "Centroids\n1-SE (post hoc)")
  mm <- names(MODELS)

  # (1) leaky vs nested CV AUC by model
  png(file.path(figdir, "fig1_leaky_vs_nested_cv_auc.png"), 1800, 1150, res = 170)
  par(mar = c(4.5, 19, 3, 1))
  aucs <- lapply(mm, function(m) { d <- out$cv[[m]]$by_rep; d$AUC[d$threshold == "0.5"] })
  lo <- min(unlist(aucs)) - 0.02
  boxplot(rev(aucs), horizontal = TRUE, names = rev(MODELS), las = 1, col = rev(cols[mm]), border = "grey20",
          ylim = c(lo, 1), xlab = "Nested-CV AUC on the 38 training patients (20 repeats; spread = range across repeats)",
          main = "Leaky vs nested cross-validation")
  points(max(out$caret_original$cv_AUC), length(mm) + 0.4, pch = 8, cex = 1.3)
  text(max(out$caret_original$cv_AUC), length(mm) + 0.4, "original caret CV (all k tied)", pos = 2, cex = 0.8)
  dev.off()

  # (2) test ROC curves
  png(file.path(figdir, "fig2_test_roc_curves.png"), 1250, 1250, res = 170)
  par(pty = "s"); first <- TRUE
  rocm <- c("knn_honest", "knn_t", "enet", "nsc_1se")
  for (m in rocm) {
    r <- roc(y_test, out$test_p[[m]], levels = c("ALL", "AML"), direction = "<", quiet = TRUE)
    plot(r, legacy.axes = TRUE, col = cols[m], lwd = 3, add = !first,
         main = "Independent test set (n = 34): ROC curves",
         xlab = "1 - Specificity (false positive rate)", ylab = "Sensitivity (AML recall)")
    first <- FALSE
  }
  lg <- sapply(rocm, function(m) sprintf("%s: AUC %s", MODELS[[m]], out$test$AUC_bootstrap_CI[out$test$model == MODELS[[m]] & out$test$threshold == "0.5"]))
  legend("bottomright", lg, col = cols[rocm], lwd = 3, bty = "n", cex = 0.68)
  dev.off()

  # (3) AML recall at 0.5 vs tuned threshold (test set, exact 95% CI)
  png(file.path(figdir, "fig3_aml_recall_thresholds.png"), 1900, 1050, res = 170)
  par(mar = c(5.5, 4.5, 4, 1))
  get <- function(col) sapply(mm, function(m) sapply(c("0.5", "Youden (training inner CV)"), function(th)
    out$test[[col]][out$test$model == MODELS[[m]] & out$test$threshold == th]))
  M <- get("sens_est"); LO <- get("sens_lo"); HI <- get("sens_hi")
  bp <- barplot(M, beside = TRUE, ylim = c(0, 1.08), col = c("grey75", "darkgreen"), names.arg = short[mm], cex.names = 0.72,
                ylab = "AML recall on the test set (14 AML)", main = "AML recall: default 0.5 vs cut-off tuned in training CV\n(error bars: exact 95% CI)")
  arrows(bp, LO, bp, HI, angle = 90, code = 3, length = 0.03)
  legend("top", c("cut-off 0.5", "Youden cut-off from inner CV"), fill = c("grey75", "darkgreen"), bty = "n", cex = 0.85,
         horiz = TRUE, inset = -0.03, xpd = TRUE)
  dev.off()

  # (4) permutation null
  png(file.path(figdir, "fig4_permutation_null.png"), 1500, 1000, res = 170)
  par(mar = c(4.5, 4.5, 3, 1))
  hist(out$perm$null, breaks = 30, xlim = c(min(out$perm$null, 0.2), 1), col = "grey80", border = "white",
       xlab = "Mean nested-CV AUC (elastic net) with permuted training labels",
       main = sprintf("Permutation null (%d permutations); observed AUC %.3f, p = %.3f",
                      length(out$perm$null), out$perm$observed, out$perm$p))
  abline(v = out$perm$observed, col = "firebrick", lwd = 3)
  dev.off()

  # (5) gene selection frequency
  png(file.path(figdir, "fig5_gene_selection_frequency.png"), 2400, 1250, res = 170)
  par(mfrow = c(1, 3), mar = c(4.5, 10, 3, 1), oma = c(2.5, 0, 0, 0))
  for (m in c("enet", "nsc", "nsc_1se")) {
    d <- head(out$freq[out$freq$model == MODELS[[m]], ], 20)
    lab <- paste0(d$gene_symbol, ifelse(d$in_original_top10, " *", ""))
    barplot(rev(d$selection_frequency), names.arg = rev(lab), horiz = TRUE, las = 1, xlim = c(0, 1), cex.names = 0.75,
            col = rev(ifelse(d$higher_in == "AML", "firebrick", "steelblue")),
            xlab = "Selected in share of 100 subsamples",
            main = sprintf("%s\n(median %g probes per fit)", MODELS[[m]], d$median_genes_selected_per_fit[1]), cex.main = 0.95)
  }
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
  legend("bottom", c("higher in AML", "higher in ALL", "* = in original top-10 fold-change list"),
         pch = c(15, 15, NA), col = c("firebrick", "steelblue", NA), pt.cex = 2, bty = "n", horiz = TRUE, cex = 0.9)
  dev.off()

  # (6) panel-size curve (version 2)
  png(file.path(figdir, "fig6_panel_size_curve.png"), 1600, 1050, res = 170)
  par(mar = c(4.5, 4.5, 4, 1))
  pz <- out$panel
  plot(pz$genes, pz$cv_balacc_mean, log = "x", type = "b", pch = 19, lwd = 2, col = "darkblue", ylim = c(0.4, 1.02),
       xaxt = "n", xlab = "Number of genes in the panel (log scale)", ylab = "Score",
       main = "How many genes? Nested CV on the training set (choice made here)\nvs test set (descriptive only)")
  axis(1, at = pz$genes)
  arrows(pz$genes, pz$cv_balacc_lo, pz$genes, pz$cv_balacc_hi, angle = 90, code = 3, length = 0.03, col = "darkblue")
  lines(pz$genes, pz$cv_AUC_mean, type = "b", pch = 17, lwd = 2, col = "darkorange")
  lines(pz$genes, pz$test_acc_est, type = "b", pch = 1, lwd = 1, lty = 2, col = "grey40")
  legend("bottomright", c("CV balanced accuracy at 0.5 (bars: range across repeats)", "CV AUC",
                          "test accuracy at 0.5 (34 patients, not used for any choice)"),
         col = c("darkblue", "darkorange", "grey40"), pch = c(19, 17, 1), lty = c(1, 1, 2), lwd = c(2, 2, 1), bty = "n", cex = 0.8)
  dev.off()

  # (7) hospital (version 2)
  png(file.path(figdir, "fig7_test_probability_by_hospital.png"), 1600, 1050, res = 170)
  par(mar = c(5, 4.5, 4, 1))
  p <- out$test_p$enet
  grp <- factor(paste(hospital_test, y_test, sep = "\n"),
                levels = c("DFCI\nALL", "St-Jude\nALL", "CALGB\nAML", "St-Jude\nAML", "CCG\nAML"))
  set.seed(1)                                            # jitter only
  xx <- as.integer(grp) + runif(length(p), -0.15, 0.15)
  plot(xx, p, xaxt = "n", xlim = c(0.5, 5.5), ylim = c(0, 1), pch = ifelse(source_test == "PB", 1, 19), cex = 1.2,
       col = c("steelblue", "steelblue", "firebrick", "firebrick", "firebrick")[as.integer(grp)],
       xlab = "", ylab = "Elastic net P(AML), test set",
       main = "Test probabilities by hospital\nTraining: all ALL from DFCI, all AML from CALGB")
  axis(1, at = 1:5, labels = levels(grp), padj = 0.5, cex.axis = 0.85)
  enet_cut <- out$test$cut_off[out$test$model == MODELS[["enet"]] & out$test$threshold != "0.5"]
  abline(h = 0.5, lty = 2); abline(h = enet_cut, lty = 3, col = "darkgreen")
  abline(v = 2.5, col = "grey80")
  legend("left", c("bone marrow", "peripheral blood", "cut-off 0.5", "Youden cut-off"), pch = c(19, 1, NA, NA),
         lty = c(NA, NA, 2, 3), col = c("grey30", "grey30", "black", "darkgreen"), bty = "n", cex = 0.8)
  mtext("hospitals in training: DFCI, CALGB | new: St-Jude, CCG", side = 1, line = 3.5, cex = 0.8)
  dev.off()
}


# =============================================================================
# STEP 6: RUN (seed 2026 = main results; seed 2028 = sensitivity check)
# =============================================================================
args <- commandArgs(trailingOnly = TRUE)
seeds <- if (length(args)) as.integer(args) else c(2026, 2028)
for (s in seeds) {
  cat("\n================ SEED", s, "================\n")
  base <- if (QUICK) "results_quick" else "results"
  res <- run_analysis(s, if (s == 2026) base else file.path(base, paste0("seed", s)),
                      do_figures = (s == 2026 && !QUICK))
}
if (!QUICK) writeLines(capture.output(sessionInfo()), "results/sessionInfo.txt")
