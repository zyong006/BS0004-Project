###############################################################################
#  APPENDIX: more rigorous machine learning on the Golub (1999) ALL vs AML data
#
#  This script reproduces every appendix result from a clean R session:
#     Rscript appendix_advanced_ml.R          (about 15-25 minutes on 24 cores)
#
#  What it adds to the main project script (which is NOT modified):
#    A. Nested cross-validation on the 38 training patients
#       (outer: stratified 5-fold x 20 repeats; inner: stratified 5-fold)
#    B. Three models through the same nested procedure:
#         1. KNN as in our script (top 50 genes by |log2 FC|), honest and leaky
#         2. Elastic net logistic regression  (PRE-DECLARED PRIMARY MODEL)
#         3. Nearest shrunken centroids (pamr)
#    C. Decision threshold: default 0.5 and Youden's J on inner-fold
#       out-of-fold probabilities
#    D. Independent test set (34 patients) used ONCE per model
#    E. Bootstrap 95% CIs on the test set; label-permutation test
#    F. Gene selection stability (100 stratified 80% subsamples)
#    G. Bone marrow vs peripheral blood breakdown
#    H. Seed sensitivity (whole analysis rerun with seed 2028)
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

N_CORES  <- max(1, min(24, detectCores() - 2))
N_REP    <- 20      # outer cross-validation repeats
N_OUTER  <- 5       # outer folds
N_INNER  <- 5       # inner folds
N_BOOT   <- 2000    # bootstrap resamples on the test set
N_PERM   <- 200     # label permutations (brief asks for at least 100)
N_SUB    <- 100     # 80% subsamples for gene stability
KNN_GENES <- 50     # as in our script
K_MAX    <- 15      # as in our script: odd k from 1 to 15
ALPHA    <- 0.5     # elastic net mixing
QUICK    <- Sys.getenv("QUICK") == "1"   # smoke test only: tiny settings, separate output folder
if (QUICK) { N_REP <- 2; N_BOOT <- 50; N_PERM <- 4; N_SUB <- 4 }

ORIGINAL_TOP10 <- c("X82240_rna1_at", "M84526_at", "X76223_s_at", "M27891_at",
                    "M30703_s_at", "L33930_s_at", "D88270_at", "M57731_s_at",
                    "M20203_s_at", "M27783_s_at")


# =============================================================================
# STEP 1: DATA AND PREPROCESSING (identical to the main script)
# =============================================================================
data(Golub_Train)
data(Golub_Test)

X_train <- t(exprs(Golub_Train))      # 38 x 7129
X_test  <- t(exprs(Golub_Test))       # 34 x 7129
y_train <- factor(Golub_Train$ALL.AML, levels = c("ALL", "AML"))
y_test  <- factor(Golub_Test$ALL.AML,  levels = c("ALL", "AML"))
source_test <- as.character(Golub_Test$BM.PB)

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
rank_fold_change <- function(Xraw, y) {
  genes <- variation_filter(Xraw)
  fc <- abs(log2(colMeans(Xraw[y == "AML", genes, drop = FALSE]) /
                 colMeans(Xraw[y == "ALL", genes, drop = FALSE])))
  names(sort(fc, decreasing = TRUE))
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


# -----------------------------------------------------------------------------
# MODEL 1: KNN (top 50 genes by |log2 FC|, odd k)
#   genes = NULL  -> honest: ranking redone inside the fitting rows and again
#                    inside every inner fold
#   genes = <vec> -> leaky reproduction: genes fixed in advance from all 38
#   k grid: odd k <= 15 AND k < 2 x (AML patients in the smallest inner fitting
#   set), so that AML can always win a majority vote.
#   k choice: highest inner out-of-fold AUC; ties -> lowest Brier score ->
#   smallest k (declared in advance; the original relied on caret's tie-break).
# -----------------------------------------------------------------------------
fit_knn <- function(Xr, Xl, y, inner, genes = NULL) {
  min_aml <- min(sapply(inner, function(v) sum(y[-v] == "AML")))
  ks <- seq(1, K_MAX, by = 2); ks <- ks[ks < 2 * min_aml]
  oof <- matrix(NA, length(y), length(ks))
  for (v in inner) {
    g  <- if (is.null(genes)) rank_fold_change(Xr[-v, , drop = FALSE], y[-v])[1:KNN_GENES] else genes
    z  <- zfit(Xl[-v, g, drop = FALSE])
    oof[v, ] <- knn_votes(zapply(Xl[-v, g, drop = FALSE], z), y[-v],
                          zapply(Xl[v, g, drop = FALSE], z), ks)
  }
  aucs  <- apply(oof, 2, function(p) as.numeric(auc(y, p, levels = c("ALL", "AML"), direction = "<", quiet = TRUE)))
  brier <- apply(oof, 2, function(p) mean((p - (y == "AML"))^2))
  cand  <- which(abs(aucs - max(aucs)) < 1e-12)
  best  <- cand[order(brier[cand], ks[cand])][1]
  g  <- if (is.null(genes)) rank_fold_change(Xr, y)[1:KNN_GENES] else genes
  z  <- zfit(Xl[, g, drop = FALSE])
  list(model = "KNN", genes = g, z = z, Ztr = zapply(Xl[, g, drop = FALSE], z), ytr = y,
       k = ks[best], n_k_tied = length(cand), inner_auc = aucs[best],
       cut = youden_cut(y, oof[, best]))
}
predict_knn <- function(fit, Xl) as.vector(knn_votes(fit$Ztr, fit$ytr, zapply(Xl[, fit$genes, drop = FALSE], fit$z), fit$k))


# -----------------------------------------------------------------------------
# MODEL 2: ELASTIC NET logistic regression (primary model)
#   genes: variation filter on the fitting rows; glmnet standardises each gene
#   with fitting-row statistics. alpha = 0.5. Class weights = n / (2 n_class).
#   lambda: inner stratified CV (binomial deviance, lambda.min).
#   Threshold: Youden on the inner out-of-fold probabilities at lambda.min.
#   Note: inner folds share the filtered gene list of their fitting rows; the
#   filter uses no labels, so this cannot carry label information.
# -----------------------------------------------------------------------------
fit_enet <- function(Xr, Xl, y, inner) {
  g <- variation_filter(Xr)
  w <- ifelse(y == "AML", length(y) / (2 * sum(y == "AML")), length(y) / (2 * sum(y == "ALL")))
  foldid <- integer(length(y)); for (i in seq_along(inner)) foldid[inner[[i]]] <- i
  cv <- cv.glmnet(Xl[, g, drop = FALSE], y, family = "binomial", alpha = ALPHA, weights = w,
                  foldid = foldid, type.measure = "deviance", keep = TRUE, standardize = TRUE)
  li  <- which(cv$lambda == cv$lambda.min)
  # glmnet 4.x stores binomial out-of-fold fits as log-odds (checked against a
  # manual fold refit), so convert them to probabilities before the Youden step
  oof <- plogis(cv$fit.preval[, li])
  stopifnot(!anyNA(oof))
  b <- as.matrix(coef(cv$glmnet.fit, s = cv$lambda.min))[-1, 1]
  list(model = "Elastic net", genes = g, fit = cv$glmnet.fit, lambda = cv$lambda.min,
       selected = names(b)[b != 0], coef = b[b != 0], cut = youden_cut(y, oof))
}
predict_enet <- function(fit, Xl) as.vector(predict(fit$fit, Xl[, fit$genes, drop = FALSE], s = fit$lambda, type = "response"))


# -----------------------------------------------------------------------------
# MODEL 3: NEAREST SHRUNKEN CENTROIDS (pamr)
#   genes: variation filter on the fitting rows; pamr standardises internally.
#   Equal class priors (0.5, 0.5). Shrinkage threshold: lowest inner CV error;
#   ties -> the larger threshold (fewer genes). Cut-off: Youden on inner
#   out-of-fold posterior probabilities at that shrinkage threshold.
# -----------------------------------------------------------------------------
quiet <- function(expr) { out <- NULL; invisible(capture.output(out <- expr)); out }
fit_nsc <- function(Xr, Xl, y, inner) {
  g <- variation_filter(Xr)
  dat <- list(x = t(Xl[, g, drop = FALSE]), y = y, geneid = g, genenames = g)
  fit <- quiet(pamr.train(dat, prior = c(0.5, 0.5)))
  cv  <- quiet(pamr.cv(fit, dat, folds = inner))
  ti  <- max(which(abs(cv$error - min(cv$error)) < 1e-12))
  thr <- cv$threshold[ti]
  oof <- cv$prob[, 2, ti]                 # column 2 = AML (levels ALL, AML)
  nz  <- quiet(pamr.predict(fit, dat$x, threshold = thr, type = "nonzero"))
  list(model = "Shrunken centroids", genes = g, fit = fit, threshold = thr,
       selected = g[nz], cut = youden_cut(y, oof))
}
predict_nsc <- function(fit, Xl) as.vector(quiet(pamr.predict(fit$fit, t(Xl[, fit$genes, drop = FALSE]),
                                                              threshold = fit$threshold, type = "posterior"))[, 2])


# one entry point for all four procedures; receives ONLY fitting rows
fit_model <- function(m, Xr, Xl, y, inner, leaky_genes = NULL) {
  switch(m,
    knn_leaky  = fit_knn(Xr, Xl, y, inner, genes = leaky_genes),
    knn_honest = fit_knn(Xr, Xl, y, inner),
    enet       = fit_enet(Xr, Xl, y, inner),
    nsc        = fit_nsc(Xr, Xl, y, inner))
}
predict_model <- function(m, fit, Xl) switch(m, knn_leaky = , knn_honest = predict_knn(fit, Xl),
                                             enet = predict_enet(fit, Xl), nsc = predict_nsc(fit, Xl))

MODELS <- c(knn_leaky = "KNN, leaky (genes ranked on all 38)", knn_honest = "KNN, honest (genes ranked in fold)",
            enet = "Elastic net (primary)", nsc = "Shrunken centroids")

# performance at a cut-off (AML = positive); cut may differ per patient
perf <- function(y, p, cut) {
  pred <- p >= cut
  TP <- sum(pred & y == "AML"); FN <- sum(!pred & y == "AML")
  TN <- sum(!pred & y == "ALL"); FP <- sum(pred & y == "ALL")
  a  <- if (length(unique(y)) == 2) as.numeric(auc(y, p, levels = c("ALL", "AML"), direction = "<", quiet = TRUE)) else NA
  c(AUC = a, sensitivity = TP / (TP + FN), specificity = TN / (TN + FP),
    accuracy = (TP + TN) / length(y), balanced_accuracy = (TP / (TP + FN) + TN / (TN + FP)) / 2,
    TP = TP, FN = FN, TN = TN, FP = FP)
}


# =============================================================================
# STEP 3: NESTED CROSS-VALIDATION (one call = one model x one set of labels)
# =============================================================================
make_plan <- function(y) {
  # all random fold assignments are drawn here, in the main process, so the
  # results do not depend on the number of cores
  lapply(1:N_REP, function(r) {
    outer <- strat_folds(y, N_OUTER)
    list(outer = outer, inner = lapply(outer, function(v) strat_folds(y[-v], N_INNER)))
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


# =============================================================================
# STEP 4: THE WHOLE ANALYSIS FOR ONE SEED
# =============================================================================
run_analysis <- function(seed, outdir, do_figures = TRUE) {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  set.seed(seed)
  log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), ..., "\n")
  out <- list(seed = seed)

  # ---- A1. the original leaky estimate, reproduced exactly with caret -------
  # genes ranked once on all 38 patients, z-scored on all 38, then 5-fold CV
  ranked_all <- rank_fold_change(X_train, y_train)
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

  # ---- A2. nested CV for all four procedures (same folds for every model) ----
  set.seed(seed)
  plan <- make_plan(y_train)
  out$cv <- list()
  for (m in names(MODELS)) {
    t0 <- Sys.time()
    out$cv[[m]] <- nested_cv(m, y_train, plan, leaky_genes = if (m == "knn_leaky") g50 else NULL)
    log_msg("nested CV done:", m, round(difftime(Sys.time(), t0, units = "mins"), 1), "min")
  }
  cv_summary <- do.call(rbind, lapply(names(MODELS), function(m) {
    d <- out$cv[[m]]$by_rep
    do.call(rbind, lapply(split(d, d$threshold), function(s) {
      st <- function(x) sprintf("%.3f (%.3f-%.3f)", mean(x), quantile(x, 0.025), quantile(x, 0.975))
      data.frame(model = MODELS[[m]], threshold = s$threshold[1],
                 AUC = st(s$AUC), sensitivity = st(s$sensitivity), specificity = st(s$specificity),
                 accuracy = st(s$accuracy), balanced_accuracy = st(s$balanced_accuracy),
                 AUC_mean = mean(s$AUC), AUC_sd = sd(s$AUC))
    }))
  }))
  rownames(cv_summary) <- NULL
  out$cv_summary <- cv_summary
  write.csv(cv_summary, file.path(outdir, "A2_nested_cv_performance.csv"), row.names = FALSE)
  write.csv(do.call(rbind, lapply(names(MODELS), function(m) cbind(model = MODELS[[m]], out$cv[[m]]$by_rep))),
            file.path(outdir, "A2_nested_cv_by_repeat.csv"), row.names = FALSE)
  # caret (and so our original script) reports the AUC averaged over the 5
  # validation folds of about 8 patients each; the table above pools every
  # out-of-fold prediction first. Show both, so the two effects can be told apart.
  fold_avg <- do.call(rbind, lapply(names(MODELS), function(m) {
    o <- out$cv[[m]]$oof
    fa <- sapply(split(o, list(o$rep, o$fold)), function(d)
      as.numeric(auc(d$y, d$p, levels = c("ALL", "AML"), direction = "<", quiet = TRUE)))
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
    data.frame(model = MODELS[[m]], fits = nrow(o),
               k_values = if (all(is.na(o$k))) "n/a" else paste(names(table(o$k)), table(o$k), sep = ":", collapse = " "),
               fits_with_tied_k = if (all(is.na(o$n_k_tied))) NA else sum(o$n_k_tied > 1),
               genes_median = median(o$n_genes), genes_range = paste(range(o$n_genes), collapse = "-"),
               youden_cut_median = round(median(o$cut), 3), youden_cut_range = paste(round(range(o$cut), 3), collapse = " to "))
  }))
  out$tuning <- tun
  write.csv(tun, file.path(outdir, "A3_nested_cv_tuning.csv"), row.names = FALSE)

  # ---- D. refit on all 38, predict the test set ONCE per model ---------------
  set.seed(seed)
  inner_all <- strat_folds(y_train, N_INNER)
  out$final <- list(); out$test_p <- list()
  for (m in names(MODELS)) {
    out$final[[m]]  <- fit_model(m, X_train, L_train, y_train, inner_all, leaky_genes = g50)
    out$test_p[[m]] <- predict_model(m, out$final[[m]], L_test)
  }
  final_desc <- do.call(rbind, lapply(names(MODELS), function(m) {
    f <- out$final[[m]]
    data.frame(model = MODELS[[m]],
               hyperparameter = switch(m, knn_leaky = , knn_honest = sprintf("k = %d (%d k values tied on inner AUC)", f$k, f$n_k_tied),
                                       enet = sprintf("lambda = %.4g", f$lambda), nsc = sprintf("shrinkage threshold = %.3f", f$threshold)),
               genes_used = if (!is.null(f$selected)) length(f$selected) else length(f$genes),
               youden_cut = round(f$cut, 3))
  }))
  out$final_desc <- final_desc
  write.csv(final_desc, file.path(outdir, "D1_final_models.csv"), row.names = FALSE)

  # ---- E1. stratified bootstrap 95% CIs on the test set ----------------------
  set.seed(seed)
  ia <- which(y_test == "AML"); il <- which(y_test == "ALL")
  boot_idx <- replicate(N_BOOT, c(il[sample.int(length(il), replace = TRUE)], ia[sample.int(length(ia), replace = TRUE)]))
  boot_ci <- function(p, cut, idx_set = boot_idx, yy = y_test) {
    est <- perf(yy, p, cut)
    B <- apply(idx_set, 2, function(b) perf(yy[b], p[b], cut)[1:5])
    ci <- apply(B, 1, quantile, c(0.025, 0.975), na.rm = TRUE)
    list(est = est, ci = ci)
  }
  fmt <- function(e, ci, nm) sprintf("%.3f (%.3f-%.3f)", e[nm], ci[1, nm], ci[2, nm])
  test_tab <- list(); conf_tab <- list()
  for (m in names(MODELS)) for (th in c("0.5", "Youden (training inner CV)")) {
    cut <- if (th == "0.5") 0.5 else out$final[[m]]$cut
    b <- boot_ci(out$test_p[[m]], cut)
    test_tab[[length(test_tab) + 1]] <- data.frame(model = MODELS[[m]], threshold = th, cut_off = round(cut, 3),
      AUC = fmt(b$est, b$ci, "AUC"), sensitivity_AML_recall = fmt(b$est, b$ci, "sensitivity"),
      specificity = fmt(b$est, b$ci, "specificity"), accuracy = fmt(b$est, b$ci, "accuracy"),
      sens_est = b$est["sensitivity"], sens_lo = b$ci[1, "sensitivity"], sens_hi = b$ci[2, "sensitivity"],
      AUC_est = b$est["AUC"], AUC_lo = b$ci[1, "AUC"], AUC_hi = b$ci[2, "AUC"])
    conf_tab[[length(conf_tab) + 1]] <- data.frame(model = MODELS[[m]], threshold = th,
      TP = b$est["TP"], FN = b$est["FN"], TN = b$est["TN"], FP = b$est["FP"])
  }
  out$test <- do.call(rbind, test_tab); rownames(out$test) <- NULL
  out$confusion <- do.call(rbind, conf_tab); rownames(out$confusion) <- NULL
  write.csv(out$test, file.path(outdir, "D2_test_performance_bootstrap_ci.csv"), row.names = FALSE)
  write.csv(out$confusion, file.path(outdir, "D3_test_confusion_matrices.csv"), row.names = FALSE)
  # per-patient test probabilities (for transparency)
  write.csv(data.frame(patient = rownames(L_test), class = y_test, source = source_test,
                       hospital = as.character(Golub_Test$Source),
                       sapply(names(MODELS), function(m) round(out$test_p[[m]], 3))),
            file.path(outdir, "D4_test_probabilities.csv"), row.names = FALSE)
  # which test patients were misclassified, and where did they come from?
  # (training samples came from two hospitals only: DFCI and CALGB)
  wrong <- do.call(rbind, lapply(names(MODELS), function(m) do.call(rbind, lapply(c("0.5", "Youden (training inner CV)"), function(th) {
    cut <- if (th == "0.5") 0.5 else out$final[[m]]$cut
    i <- which((out$test_p[[m]] >= cut) != (y_test == "AML"))
    if (!length(i)) return(NULL)
    data.frame(model = MODELS[[m]], threshold = th, patient = rownames(L_test)[i], class = y_test[i],
               source = source_test[i], hospital = as.character(Golub_Test$Source)[i],
               hospital_in_training = as.character(Golub_Test$Source)[i] %in% unique(as.character(Golub_Train$Source)),
               p_AML = round(out$test_p[[m]][i], 3))
  }))))
  out$wrong <- wrong
  write.csv(wrong, file.path(outdir, "D5_misclassified_test_patients.csv"), row.names = FALSE)
  log_msg("test set evaluated")

  # ---- G. bone marrow vs peripheral blood ------------------------------------
  src_tab <- list()
  for (m in names(MODELS)) for (src in c("BM", "PB")) for (th in c("0.5", "Youden (training inner CV)")) {
    idx <- which(source_test == src)
    cut <- if (th == "0.5") 0.5 else out$final[[m]]$cut
    ys <- y_test[idx]; ps <- out$test_p[[m]][idx]
    a <- which(ys == "AML"); l <- which(ys == "ALL")
    bi <- replicate(N_BOOT, c(l[sample.int(length(l), replace = TRUE)], a[sample.int(length(a), replace = TRUE)]))
    b <- boot_ci(ps, cut, bi, ys)
    src_tab[[length(src_tab) + 1]] <- data.frame(model = MODELS[[m]], source = src, threshold = th,
      n = length(idx), n_ALL = length(l), n_AML = length(a), errors = b$est["FN"] + b$est["FP"],
      AUC = fmt(b$est, b$ci, "AUC"), sensitivity = fmt(b$est, b$ci, "sensitivity"),
      specificity = fmt(b$est, b$ci, "specificity"), accuracy = fmt(b$est, b$ci, "accuracy"))
  }
  out$source <- do.call(rbind, src_tab); rownames(out$source) <- NULL
  write.csv(out$source, file.path(outdir, "G_bone_marrow_vs_blood.csv"), row.names = FALSE)

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
    list(s = s, inner = strat_folds(y_train[s], N_INNER))
  })
  sel <- mclapply(subs, function(ss) {
    list(enet = fit_enet(X_train[ss$s, ], L_train[ss$s, ], y_train[ss$s], ss$inner)$selected,
         nsc  = fit_nsc(X_train[ss$s, ], L_train[ss$s, ], y_train[ss$s], ss$inner)$selected)
  }, mc.cores = N_CORES, mc.preschedule = FALSE)
  annot <- read.delim(gzfile("annotation/GPL80.annot.gz"), skip = grep("^ID\t", readLines(gzfile("annotation/GPL80.annot.gz"), n = 200)) - 1,
                      quote = "", check.names = FALSE, stringsAsFactors = FALSE, fill = TRUE)
  sym <- setNames(annot[["Gene symbol"]], annot$ID)
  freq_tab <- do.call(rbind, lapply(c("enet", "nsc"), function(m) {
    f <- sort(table(unlist(lapply(sel, `[[`, m))), decreasing = TRUE) / N_SUB
    pr <- names(f)
    dirn <- ifelse(colMeans(L_train[y_train == "AML", pr, drop = FALSE]) > colMeans(L_train[y_train == "ALL", pr, drop = FALSE]), "AML", "ALL")
    data.frame(model = MODELS[[m]], probe = pr, gene_symbol = ifelse(is.na(sym[pr]) | sym[pr] == "", "(none)", sym[pr]),
               selection_frequency = as.numeric(f), higher_in = dirn,
               in_original_top10 = pr %in% ORIGINAL_TOP10,
               fold_change_rank_all38 = match(pr, ranked_all),
               median_genes_selected_per_fit = median(sapply(sel, function(s) length(s[[m]]))))
  }))
  rownames(freq_tab) <- NULL
  out$freq <- freq_tab
  write.csv(freq_tab, file.path(outdir, "F_gene_selection_frequency.csv"), row.names = FALSE)
  # how often were the original top-10 genes selected?
  top10_tab <- data.frame(probe = ORIGINAL_TOP10, gene_symbol = sym[ORIGINAL_TOP10],
    enet_frequency = sapply(ORIGINAL_TOP10, function(p) mean(sapply(sel, function(s) p %in% s$enet))),
    nsc_frequency  = sapply(ORIGINAL_TOP10, function(p) mean(sapply(sel, function(s) p %in% s$nsc))))
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
    # final model: the test set is never passed to fit_model at all; check that
    # predictions on the training set do not change when the test set is noise
    L_test_bad <- matrix(runif(length(L_test), 2, 4.2), nrow(L_test), dimnames = dimnames(L_test))
    p1 <- predict_model(m, out$final[[m]], L_train); f2 <- fit_model(m, X_train, L_train, y_train, inner_all, g50)
    canary[[length(canary) + 1]] <- data.frame(check = "test set replaced by noise", model = MODELS[[m]],
      identical_fit = same_fit(out$final[[m]], f2),
      identical_predictions_on_training_rows = identical(p1, predict_model(m, f2, L_train)))
    invisible(L_test_bad)
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
  cols <- c(knn_leaky = "grey55", knn_honest = "steelblue", enet = "firebrick", nsc = "darkorange")

  # (1) leaky vs nested CV AUC by model
  png(file.path(figdir, "fig1_leaky_vs_nested_cv_auc.png"), 1700, 1000, res = 170)
  par(mar = c(4.5, 17, 3, 1))
  aucs <- lapply(names(MODELS), function(m) { d <- out$cv[[m]]$by_rep; d$AUC[d$threshold == "0.5"] })
  lo <- min(unlist(aucs), out$caret_original$cv_AUC) - 0.02
  boxplot(rev(aucs), horizontal = TRUE, names = rev(MODELS), las = 1, col = rev(cols), border = "grey20",
          ylim = c(lo, 1), xlab = "Cross-validated AUC on the 38 training patients (20 repeats of 5-fold)",
          main = "Leaky vs nested cross-validation")
  points(max(out$caret_original$cv_AUC), 4.5, pch = 8, cex = 1.3)
  text(max(out$caret_original$cv_AUC), 4.5, "original caret CV (all k tied)", pos = 2, cex = 0.8)
  dev.off()

  # (2) test ROC curves for the three models
  png(file.path(figdir, "fig2_test_roc_curves.png"), 1200, 1200, res = 170)
  par(pty = "s")
  first <- TRUE
  for (m in c("knn_honest", "enet", "nsc")) {
    r <- roc(y_test, out$test_p[[m]], levels = c("ALL", "AML"), direction = "<", quiet = TRUE)
    plot(r, legacy.axes = TRUE, col = cols[m], lwd = 3, add = !first,
         main = "Independent test set (n = 34): ROC curves",
         xlab = "1 - Specificity (false positive rate)", ylab = "Sensitivity (AML recall)")
    first <- FALSE
  }
  lg <- sapply(c("knn_honest", "enet", "nsc"), function(m) { t <- out$test[out$test$model == MODELS[[m]] & out$test$threshold == "0.5", ]
    sprintf("%s: AUC %s", MODELS[[m]], t$AUC) })
  legend("bottomright", lg, col = cols[c("knn_honest", "enet", "nsc")], lwd = 3, bty = "n", cex = 0.75)
  dev.off()

  # (3) AML recall at 0.5 vs tuned threshold (test set, 95% bootstrap CI)
  png(file.path(figdir, "fig3_aml_recall_thresholds.png"), 1600, 1000, res = 170)
  par(mar = c(4.5, 4.5, 3, 1))
  mm <- names(MODELS)
  M  <- sapply(mm, function(m) sapply(c("0.5", "Youden (training inner CV)"), function(th)
    out$test$sens_est[out$test$model == MODELS[[m]] & out$test$threshold == th]))
  LO <- sapply(mm, function(m) sapply(c("0.5", "Youden (training inner CV)"), function(th)
    out$test$sens_lo[out$test$model == MODELS[[m]] & out$test$threshold == th]))
  HI <- sapply(mm, function(m) sapply(c("0.5", "Youden (training inner CV)"), function(th)
    out$test$sens_hi[out$test$model == MODELS[[m]] & out$test$threshold == th]))
  bp <- barplot(M, beside = TRUE, ylim = c(0, 1.08), col = c("grey75", "darkgreen"),
                names.arg = c("KNN leaky", "KNN honest", "Elastic net", "Shrunken\ncentroids"),
                ylab = "AML recall on the test set (14 AML)", main = "AML recall: default 0.5 vs threshold tuned in training CV")
  arrows(bp, LO, bp, HI, angle = 90, code = 3, length = 0.04)
  legend("top", c("cut-off 0.5", "Youden cut-off from inner CV"), fill = c("grey75", "darkgreen"), bty = "n", cex = 0.85, horiz = TRUE, inset = -0.02, xpd = TRUE)
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
  png(file.path(figdir, "fig5_gene_selection_frequency.png"), 1800, 1200, res = 170)
  par(mfrow = c(1, 2), mar = c(4.5, 10, 3, 1), oma = c(2.5, 0, 0, 0))
  for (m in c("enet", "nsc")) {
    # top 20 by frequency; ties ordered by the fold-change rank on all 38 patients
    d <- out$freq[out$freq$model == MODELS[[m]], ]
    d <- head(d[order(-d$selection_frequency, d$fold_change_rank_all38), ], 20)
    lab <- paste0(d$gene_symbol, ifelse(d$in_original_top10, " *", ""))
    barplot(rev(d$selection_frequency), names.arg = rev(lab), horiz = TRUE, las = 1, xlim = c(0, 1), cex.names = 0.7,
            col = rev(ifelse(d$higher_in == "AML", "firebrick", "steelblue")),
            xlab = "Selected in share of 100 subsamples", main = MODELS[[m]])
  }
  par(fig = c(0, 1, 0, 1), oma = c(0, 0, 0, 0), mar = c(0, 0, 0, 0), new = TRUE)
  plot(0, 0, type = "n", bty = "n", xaxt = "n", yaxt = "n")
  legend("bottom", c("higher in AML", "higher in ALL", "* = in original top-10 fold-change list"),
         pch = c(15, 15, NA), col = c("firebrick", "steelblue", NA), pt.cex = 2, bty = "n", horiz = TRUE, cex = 0.85)
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
