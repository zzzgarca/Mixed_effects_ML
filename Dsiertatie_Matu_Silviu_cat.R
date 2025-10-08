# ================== Packages ==================
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  dplyr, tidyr, tibble, purrr, stringr, readr,
  MCMCglmm, pROC, PRROC
)

# ================== Setup ==================
working_folder_path <- "/Users/silviumatu/Desktop/Code/R/Disertatie"
setwd(working_folder_path)
options(max.print = 9999, width = 500)
set.seed(1234)

# ================== Data ==================
GHQ_cat_df <- read.csv("PED_GHQ_categorical_data_forecast.csv")
columns_GHQ_cat_df <- read.csv("columns_PED_GHQ_categorical_data_forecast.csv")

get_cols <- function(flag) {
  columns_GHQ_cat_df %>% filter(.data[[flag]] == 1) %>% pull(column_name)
}

# Flags
GHQ_cat_outcome_cols         <- get_cols("outcomes")
GHQ_cat_participant_cols     <- get_cols("participant_id")
GHQ_cat_time_cols            <- get_cols("time")
GHQ_cat_fixed_effects_cols   <- get_cols("fixed_effects")
GHQ_cat_random_effects_cols  <- get_cols("random_effects")

# Main names
outcome_col <- GHQ_cat_outcome_cols[1]
id_col      <- GHQ_cat_participant_cols[1]
time_col    <- GHQ_cat_time_cols[1]

# ================== Types & robust outcome recode ==================
GHQ_cat_df[[id_col]] <- as.factor(GHQ_cat_df[[id_col]])
if (!is.numeric(GHQ_cat_df[[time_col]])) {
  parsed <- suppressWarnings(as.POSIXct(GHQ_cat_df[[time_col]], tz = "UTC"))
  GHQ_cat_df[[time_col]] <-
    if (all(!is.na(parsed))) as.numeric(parsed) else as.numeric(factor(GHQ_cat_df[[time_col]]))
}

to_binary_01 <- function(y){
  if (is.logical(y)) return(factor(as.integer(y), levels = c(0,1)))
  if (is.numeric(y)) return(factor(as.integer(y != 0), levels = c(0,1)))
  if (is.factor(y) && nlevels(y) == 2) {
    lev <- levels(y)
    pos_candidates <- c("1","yes","Yes","TRUE","True","true","case","positive")
    pos <- if (any(lev %in% pos_candidates)) which(lev %in% pos_candidates)[1] else 2
    return(factor(as.integer(y == lev[pos]), levels = c(0,1)))
  }
  stop("Outcome must be binary 0/1, logical, or a 2-level factor.")
}

GHQ_cat_df[[outcome_col]] <- to_binary_01(GHQ_cat_df[[outcome_col]])
GHQ_cat_df <- GHQ_cat_df %>% filter(!is.na(.data[[outcome_col]]), !is.na(.data[[id_col]]))

# ================== Helpers: metrics, threshold, splits ==================
btick <- function(x) paste0("`", x, "`")

optimal_threshold_exact <- function(y, p, beta = 1, default = 0.5) {
  y <- as.integer(as.character(y))
  p <- as.numeric(p)
  keep <- is.finite(p) & !is.na(y)
  y <- y[keep]; p <- p[keep]
  if (!length(y)) return(default)
  
  P <- sum(y); N <- length(y) - P
  if (P == 0) return(1)       # no positives -> predict all 0
  if (N == 0) return(0)       # no negatives -> predict all 1
  
  o <- order(p, decreasing = TRUE)
  p <- p[o]; y <- y[o]
  
  tp <- cumsum(y)
  fp <- cumsum(1 - y)
  prec <- tp / (tp + fp)
  rec  <- tp / P
  f <- (1 + beta^2) * prec * rec / (beta^2 * prec + rec)
  f[!is.finite(f)] <- -Inf
  
  idx <- which.max(f)
  next_p <- if (idx < length(p)) max(p[(idx+1):length(p)]) else -Inf
  thr <- if (is.finite(next_p) && next_p < p[idx]) (p[idx] + next_p)/2 else (p[idx] - .Machine$double.eps)
  
  # attach details without changing the return type
  attr(thr, "F") <- f[idx]
  attr(thr, "precision") <- prec[idx]
  attr(thr, "recall") <- rec[idx]
  thr
}



compute_metrics <- function(y, p, thr) {
  y <- as.integer(as.character(y))
  pred <- as.integer(p >= thr)
  tp <- sum(pred==1 & y==1); tn <- sum(pred==0 & y==0)
  fp <- sum(pred==1 & y==0); fn <- sum(pred==0 & y==1)
  acc <- mean(pred == y)
  prec <- ifelse(tp+fp==0, NA, tp/(tp+fp))
  rec  <- ifelse(tp+fn==0, NA, tp/(tp+fn))
  spec <- ifelse(tn+fp==0, NA, tn/(tn+fp))
  f1   <- ifelse(is.na(prec)||is.na(rec)||(prec+rec)==0, NA, 2*prec*rec/(prec+rec))
  auc  <- tryCatch(as.numeric(pROC::auc(pROC::roc(y, p, quiet = TRUE))), error=function(e) NA_real_)
  auprc <- tryCatch(PRROC::pr.curve(scores.class0 = p[y==1], scores.class1 = p[y==0], curve = FALSE)$auc.integral,
                    error=function(e) NA_real_)
  
  brier <- mean((p - y)^2, na.rm = TRUE)
  
  tibble(
    task_1_ACC = acc, task_1_AUC = auc, task_1_AUPRC = auprc, task_1_F1 = f1,
    task_1_Precision = prec, task_1_Recall = rec, task_1_Sensitivity = rec, task_1_Specificity = spec,
    task_1_Brier = brier,
  )
}

format_points <- function(dfrow, label) {
  dfrow %>%
    pivot_longer(everything(), names_to = "metric", values_to = "value") %>%
    transmute(split_type = label, metric, summary = sprintf("%.4f", value)) %>%
    arrange(metric)
}

subject_kfold_splits <- function(df, id_col, k = 5, seed = 42) {
  set.seed(seed)
  ids <- unique(df[[id_col]])
  folds <- sample(rep(seq_len(k), length.out = length(ids)))
  lapply(seq_len(k), function(i) {
    te_ids <- ids[folds == i]
    list(
      train_idx = which(!(df[[id_col]] %in% te_ids)),
      test_idx  = which( (df[[id_col]] %in% te_ids))
    )
  })
}

time_kfold_splits_by_subject <- function(df, id_col, time_col, k = 5, min_points = 4) {
  df2 <- df %>% arrange(.data[[id_col]], .data[[time_col]]) %>% mutate(.row_id = dplyr::row_number())
  by_id <- df2 %>% group_by(.data[[id_col]]) %>% group_split()
  folds <- replicate(k, list(train_idx = integer(0), test_idx = integer(0)), simplify = FALSE)
  for (g in by_id) {
    n_i <- nrow(g)
    if (n_i >= max(min_points, k)) {
      cuts <- floor(seq(0, n_i, length.out = k + 1))
      for (i in seq_len(k)) {
        idx <- if (cuts[i] < cuts[i+1]) seq.int(cuts[i] + 1, cuts[i + 1]) else integer(0)
        folds[[i]]$test_idx  <- c(folds[[i]]$test_idx,  g$.row_id[idx])
        folds[[i]]$train_idx <- c(folds[[i]]$train_idx, setdiff(g$.row_id, g$.row_id[idx]))
      }
    } else {
      for (i in seq_len(k)) folds[[i]]$train_idx <- c(folds[[i]]$train_idx, g$.row_id)
    }
  }
  folds
}

# Single-split generators (no CV)
single_split_by_participants_80_20 <- function(df, id_col, seed = 123) {
  set.seed(seed)
  ids <- unique(df[[id_col]])
  train_ids <- sample(ids, size = floor(0.8 * length(ids)))
  list(
    train_idx = which(df[[id_col]] %in% train_ids),
    test_idx  = which(!(df[[id_col]] %in% train_ids))
  )
}

single_split_time_lastX_by_subject <- function(df, id_col, time_col, min_points = 3, test_frac = 0.20) {
  df2 <- df %>% arrange(.data[[id_col]], .data[[time_col]]) %>% mutate(.row_id = dplyr::row_number())
  idx_test <- integer(0); idx_train <- integer(0)
  df2 %>% group_by(.data[[id_col]]) %>% group_walk(function(g, key){
    n_i <- nrow(g)
    if (n_i >= min_points) {
      k_i <- max(1, ceiling(test_frac * n_i))
      test_rows  <- tail(g, k_i)
      train_rows <- head(g, n_i - k_i)
    } else {
      test_rows  <- g[0, , drop=FALSE]
      train_rows <- g
    }
    idx_test  <<- c(idx_test,  test_rows$.row_id)
    idx_train <<- c(idx_train, train_rows$.row_id)
  })
  list(train_idx = idx_train, test_idx = idx_test)
}

# ================== PCA (fit on train, apply to test) ==================
choose_k <- function(pr, thresh = 0.95) {
  if (is.null(pr) || is.null(pr$sdev)) return(0L)
  ev <- pr$sdev^2; which(cumsum(ev / sum(ev)) >= thresh)[1]
}

fit_pca_train_transform <- function(train_df, test_df, fixed_only_cols, shared_cols,
                                    var_exp = 0.95, cap_fixed = Inf, cap_shared = 5L,
                                    nz_thres = 1e-12) {
  keep_numeric_nz <- function(df, cols) {
    cols[vapply(cols, function(c) {
      x <- df[[c]]
      is.numeric(x) && sum(!is.na(x)) >= 2 && stats::sd(x, na.rm = TRUE) > nz_thres
    }, logical(1))]
  }
  impute_mat <- function(mat, means) {
    if (length(means) == 0L) return(mat)
    for (j in seq_along(means)) {
      idx <- is.na(mat[, j])
      if (any(idx)) mat[idx, j] <- means[j]
    }
    mat
  }
  score_block <- function(train_df, test_df, cols, var_exp, cap, prefix) {
    cols <- keep_numeric_nz(train_df, cols)
    if (length(cols) == 0L) return(list(train=tibble(), test=tibble(), names=character(0)))
    
    tr_mat <- as.matrix(train_df[, cols, drop=FALSE])
    te_mat <- as.matrix(test_df[,  cols, drop=FALSE])
    
    # Impute TRAIN means (used for both train & test)
    tr_means <- suppressWarnings(colMeans(tr_mat, na.rm = TRUE))
    tr_mat <- impute_mat(tr_mat, tr_means)
    te_mat <- impute_mat(te_mat, tr_means)
    
    # PCA on TRAIN (after imputation)
    pr <- prcomp(tr_mat, center = TRUE, scale. = TRUE)
    
    k <- max(1L, choose_k(pr, var_exp))
    if (is.finite(cap)) k <- min(k, as.integer(cap))
    
    # Manual transform using train center/scale to avoid NA propagation
    tr_scaled <- sweep(tr_mat, 2, pr$center, "-")
    tr_scaled <- sweep(tr_scaled, 2, pr$scale,  "/")
    te_scaled <- sweep(te_mat, 2, pr$center, "-")
    te_scaled <- sweep(te_scaled, 2, pr$scale,  "/")
    
    PC_tr <- tr_scaled %*% pr$rotation[, 1:k, drop=FALSE]
    PC_te <- te_scaled %*% pr$rotation[, 1:k, drop=FALSE]
    
    nm <- paste0(prefix, seq_len(k))
    list(train = as_tibble(PC_tr, .name_repair="unique") |> setNames(nm),
         test  = as_tibble(PC_te, .name_repair="unique") |> setNames(nm),
         names = nm)
  }
  
  block_fixed  <- score_block(train_df, test_df, fixed_only_cols, var_exp, cap_fixed,  "PCf_")
  block_shared <- score_block(train_df, test_df, shared_cols,     var_exp, cap_shared, "PCs_")
  
  train_aug <- dplyr::bind_cols(train_df, block_fixed$train,  block_shared$train)
  test_aug  <- dplyr::bind_cols(test_df,  block_fixed$test,   block_shared$test)
  
  list(train = train_aug,
       test  = test_aug,
       pc_fixed_names  = block_fixed$names,
       pc_shared_names = block_shared$names)
}


# ================== MCMCglmm build & fit ==================
build_fixed_formula <- function(outcome, pc_fixed, pc_shared) {
  rhs <- if (length(c(pc_fixed, pc_shared))) paste(btick(c(pc_fixed, pc_shared)), collapse = " + ") else "1"
  as.formula(paste(btick(outcome), "~", rhs))
}
build_random_formula <- function(pc_shared, id_col) {
  if (length(pc_shared)) {
    as.formula(paste0("~ idh(1 + ", paste(btick(pc_shared), collapse = " + "), "):", btick(id_col)))
  } else {
    as.formula(paste0("~ idh(1):", btick(id_col)))
  }
}
drop_na_for_model <- function(df, outcome_col, id_col, pc_fixed_names, pc_shared_names) {
  used <- unique(c(outcome_col, id_col, pc_fixed_names, pc_shared_names))
  df %>% tidyr::drop_na(dplyr::all_of(used))
}
has_both_classes <- function(yfac) {
  levs <- levels(yfac); all(c("0","1") %in% levs) && length(unique(yfac)) == 2 && min(table(yfac)) >= 1
}
make_prior <- function(X_fixed, n_re) {
  p_fix <- ncol(X_fixed)
  B <- list(mu = rep(0, p_fix), V = diag(p_fix) * 0.05)
  G <- list(G1 = list(V = diag(n_re) * 0.2, nu = n_re + 2))
  R <- list(V = 1, fix = 1)
  list(B = B, G = G, R = R)
}

fit_predict_MCMCglmm <- function(train_df, test_df,
                                 outcome_col, id_col,
                                 pc_fixed_names, pc_shared_names,
                                 nitt = 13000, burnin = 3000, thin = 10,
                                 verbose = FALSE, seed = 42) {
  
  tr <- drop_na_for_model(train_df, outcome_col, id_col, pc_fixed_names, pc_shared_names)
  te <- drop_na_for_model(test_df,  outcome_col, id_col, pc_fixed_names, pc_shared_names)
  if (!has_both_classes(tr[[outcome_col]])) stop("TRAIN split lacks both classes; cannot fit categorical MCMC.")
  
  f_fixed  <- build_fixed_formula(outcome_col, pc_fixed_names, pc_shared_names)
  f_random <- build_random_formula(pc_shared_names, id_col)
  
  X_tr <- model.matrix(delete.response(terms(f_fixed)), data = tr)
  q_re <- 1 + length(pc_shared_names)
  prior <- make_prior(X_tr, q_re)
  
  set.seed(seed)
  fit <- MCMCglmm(
    fixed   = f_fixed,
    random  = f_random,
    family  = "categorical",
    data    = tr,
    prior   = prior,
    verbose = verbose,
    nitt    = nitt, burnin = burnin, thin = thin
  )
  
  p_tr <- tryCatch(as.numeric(predict(fit, newdata = tr, type = "response")), error = function(e) {
    beta <- colMeans(as.matrix(fit$Sol[, colnames(X_tr), drop = FALSE]))
    pnorm(as.numeric(X_tr %*% beta))
  })
  X_te <- model.matrix(delete.response(terms(f_fixed)), data = te)
  p_te <- tryCatch(as.numeric(predict(fit, newdata = te, type = "response")), error = function(e) {
    beta <- colMeans(as.matrix(fit$Sol[, colnames(X_tr), drop = FALSE]))
    pnorm(as.numeric(X_te %*% beta))
  })
  
  y_tr <- tr[[outcome_col]]
  y_te <- te[[outcome_col]]
  
  # --- choose F1-optimal threshold on TRAIN ---
  thr_opt <- optimal_threshold_exact(y_tr, p_tr)
  
  # --- metrics @ optimal threshold ---
  train_opt <- compute_metrics(y_tr, p_tr, thr_opt) %>%
    dplyr::mutate(set = "train", at = "thr_opt",
                  threshold = as.numeric(thr_opt),
                  F_opt = attr(thr_opt, "F"),
                  precision_opt = attr(thr_opt, "precision"),
                  recall_opt = attr(thr_opt, "recall"))
  
  test_opt <- compute_metrics(y_te, p_te, thr_opt) %>%
    dplyr::mutate(set = "test", at = "thr_opt",
                  threshold = as.numeric(thr_opt))
  
  # --- metrics @ fixed 0.50 threshold ---
  train_05 <- compute_metrics(y_tr, p_tr, 0.5) %>%
    dplyr::mutate(set = "train", at = "thr05", threshold = 0.5)
  
  test_05 <- compute_metrics(y_te, p_te, 0.5) %>%
    dplyr::mutate(set = "test", at = "thr05", threshold = 0.5)
  
  metrics_long <- dplyr::bind_rows(train_opt, test_opt, train_05, test_05)
  
  list(
    metrics_long = metrics_long,
    threshold    = thr_opt,
    model        = fit
  )
}

# ================== K-fold runners (with per-fold PCA) ==================
run_kfold_mcmcglmm <- function(df, folds, label,
                               fixed_effects_cols, random_effects_cols,
                               outcome_col, id_col,
                               nitt = 13000, burnin = 3000, thin = 10,
                               verbose = FALSE, seed_base = 100,
                               var_exp = 0.95, cap_fixed = Inf, cap_shared = 5L) {
  outs <- vector("list", length(folds))
  fixed_only_cols <- setdiff(fixed_effects_cols, random_effects_cols)
  shared_cols     <- intersect(fixed_effects_cols, random_effects_cols)
  
  for (i in seq_along(folds)) {
    tr <- df[folds[[i]]$train_idx, , drop = FALSE]
    te <- df[folds[[i]]$test_idx,  , drop = FALSE]
    if (nrow(te) == 0 || nrow(tr) == 0) next
    
    pca <- fit_pca_train_transform(tr, te, fixed_only_cols, shared_cols,
                                   var_exp = var_exp, cap_fixed = cap_fixed, cap_shared = cap_shared)
    
    res <- try(
      fit_predict_MCMCglmm(pca$train, pca$test, outcome_col, id_col,
                           pca$pc_fixed_names, pca$pc_shared_names,
                           nitt = nitt, burnin = burnin, thin = thin,
                           verbose = verbose, seed = seed_base + i),
      silent = TRUE
    )
    if (inherits(res, "try-error")) next
    outs[[i]] <- res$metrics_long %>%
      dplyr::mutate(fold = i, split_type = label)
  }
  dplyr::bind_rows(outs)
}

mean_ci <- function(x, conf = 0.95) {
  x <- x[is.finite(x)]
  n <- length(x); if (n == 0) return(c(mean = NA, lwr = NA, upr = NA))
  m <- mean(x); s <- sd(x); tcrit <- qt(1 - (1 - conf)/2, df = max(1, n - 1)); me <- tcrit * s / sqrt(n)
  c(mean = m, lwr = m - me, upr = m + me)
}
summarize_cv <- function(df_metrics, label) {
  df_metrics %>%
    tidyr::pivot_longer(starts_with("task_1_"), names_to = "metric", values_to = "value") %>%
    dplyr::group_by(set, at, metric) %>%
    dplyr::summarize(
      mean = mean(value, na.rm = TRUE),
      lwr  = mean_ci(value)["lwr"],
      upr  = mean_ci(value)["upr"],
      .groups = "drop"
    ) %>%
    dplyr::mutate(
      mean = sprintf("%.4f", mean),
      lwr  = sprintf("%.4f", lwr),
      upr  = sprintf("%.4f", upr),
      split_type = label,
      summary = paste0(mean, "  (95% CI ", lwr, ", ", upr, ")")
    ) %>%
    dplyr::select(split_type, set, at, metric, summary) %>%
    dplyr::arrange(split_type, set, at, metric)
}


# ================== Single-split runner (no CV) ==================
run_single_split_mcmcglmm <- function(df, split, label,
                                      fixed_effects_cols, random_effects_cols,
                                      outcome_col, id_col,
                                      nitt = 15000, burnin = 3000, thin = 10,
                                      verbose = FALSE, seed = 101,
                                      var_exp = 0.95, cap_fixed = Inf, cap_shared = 5L) {
  fixed_only_cols <- setdiff(fixed_effects_cols, random_effects_cols)
  shared_cols     <- intersect(fixed_effects_cols, random_effects_cols)
  
  tr <- df[split$train_idx, , drop = FALSE]
  te <- df[split$test_idx,  , drop = FALSE]
  
  pca <- fit_pca_train_transform(tr, te, fixed_only_cols, shared_cols,
                                 var_exp = var_exp, cap_fixed = cap_fixed, cap_shared = cap_shared)
  
  res <- fit_predict_MCMCglmm(pca$train, pca$test, outcome_col, id_col,
                              pca$pc_fixed_names, pca$pc_shared_names,
                              nitt = nitt, burnin = burnin, thin = thin,
                              verbose = verbose, seed = seed)
  
  points <- res$metrics_long %>%
    dplyr::mutate(split_type = label) %>%
    dplyr::arrange(set, at)
  
  list(points = points, threshold = res$threshold, model = res$model)
}

## ==== helpers just for pretty printing blocks ====
summarize_cv_noat <- function(df_metrics_filtered, label) {
  # uses your summarize_cv() but drops the 'at' column so we match the 4-col layout
  summarize_cv(df_metrics_filtered, label) %>%
    dplyr::select(split_type, set, metric, summary)
}

print_single_block <- function(points_tbl, block_label){
  # points_tbl is res$points (from run_single_split_mcmcglmm)
  points_tbl %>%
    tidyr::pivot_longer(starts_with("task_1_"), names_to = "metric", values_to = "value") %>%
    dplyr::group_by(split_type, set, metric) %>%
    dplyr::summarize(summary = sprintf("%.4f", value), .groups = "drop") %>%
    dplyr::arrange(split_type, set, metric) %>%
    { cat("\n---", block_label, "---\n"); print(., n = Inf) }
}

## ================== RUNNERS ==================
k <- 3  # change as needed

# K-fold CV: subject-based
folds_subject <- subject_kfold_splits(GHQ_cat_df, id_col = id_col, k = k, seed = 42)
cv_subject <- run_kfold_mcmcglmm(
  GHQ_cat_df, folds_subject, label = "subject_kfold",
  fixed_effects_cols = GHQ_cat_fixed_effects_cols,
  random_effects_cols = GHQ_cat_random_effects_cols,
  outcome_col = outcome_col, id_col = id_col,
  nitt = 13000, burnin = 3000, thin = 10, verbose = FALSE, seed_base = 500,
  var_exp = 0.95, cap_shared = 5L
)

# K-fold CV: time-based contiguous within subject
folds_time <- time_kfold_splits_by_subject(GHQ_cat_df, id_col = id_col, time_col = time_col, k = k, min_points = 2)
cv_time <- run_kfold_mcmcglmm(
  GHQ_cat_df, folds_time, label = "time_kfold",
  fixed_effects_cols = GHQ_cat_fixed_effects_cols,
  random_effects_cols = GHQ_cat_random_effects_cols,
  outcome_col = outcome_col, id_col = id_col,
  nitt = 13000, burnin = 3000, thin = 10, verbose = FALSE, seed_base = 700,
  var_exp = 0.95, cap_shared = 5L
)

## ===== CV summaries printed as two separate blocks =====

cat("\n================ CV @ TRAIN-OPTIMAL THRESHOLD ================\n")
summary_subject_opt <- summarize_cv_noat(dplyr::filter(cv_subject, at == "thr_opt"), "subject_kfold")
summary_time_opt    <- summarize_cv_noat(dplyr::filter(cv_time,    at == "thr_opt"), "time_kfold")
dplyr::bind_rows(summary_subject_opt, summary_time_opt) %>%
  dplyr::arrange(split_type, set, metric) %>%
  print(n = Inf)

cat("\n================== CV @ FIXED 0.50 THRESHOLD ==================\n")
summary_subject_05 <- summarize_cv_noat(dplyr::filter(cv_subject, at == "thr05"), "subject_kfold")
summary_time_05    <- summarize_cv_noat(dplyr::filter(cv_time,    at == "thr05"), "time_kfold")
dplyr::bind_rows(summary_subject_05, summary_time_05) %>%
  dplyr::arrange(split_type, set, metric) %>%
  print(n = Inf)

## ===== Single-split (NO CV) calls — printed as two separate blocks =====

# A) 80/20 by participants (no CV)
split_part <- single_split_by_participants_80_20(GHQ_cat_df, id_col, seed = 123)
res_part <- run_single_split_mcmcglmm(
  GHQ_cat_df, split_part, label = "subject_80_20",
  fixed_effects_cols = GHQ_cat_fixed_effects_cols,
  random_effects_cols = GHQ_cat_random_effects_cols,
  outcome_col = outcome_col, id_col = id_col,
  nitt = 15000, burnin = 3000, thin = 10, verbose = FALSE, seed = 101,
  var_exp = 0.95, cap_shared = 5L
)

print_single_block(dplyr::filter(res_part$points, at == "thr_opt"),
                   "Single split: 80/20 by participants @ optimal threshold (train)")
cat("Chosen train-optimal threshold: ", sprintf("%.3f", res_part$threshold), "\n", sep = "")

print_single_block(dplyr::filter(res_part$points, at == "thr05"),
                   "Single split: 80/20 by participants @ fixed 0.50")

# B) Last 20% by time within subject (no CV)
split_time <- single_split_time_lastX_by_subject(GHQ_cat_df, id_col, time_col, min_points = 2, test_frac = 0.20)
res_time <- run_single_split_mcmcglmm(
  GHQ_cat_df, split_time, label = "time_last20_by_subject",
  fixed_effects_cols = GHQ_cat_fixed_effects_cols,
  random_effects_cols = GHQ_cat_random_effects_cols,
  outcome_col = outcome_col, id_col = id_col,
  nitt = 15000, burnin = 3000, thin = 10, verbose = FALSE, seed = 202,
  var_exp = 0.95, cap_shared = 5L
)

print_single_block(dplyr::filter(res_time$points, at == "thr_opt"),
                   "Single split: last 20% by time within subject @ optimal threshold (train)")
cat("Chosen train-optimal threshold: ", sprintf("%.3f", res_time$threshold), "\n", sep = "")

print_single_block(dplyr::filter(res_time$points, at == "thr05"),
                   "Single split: last 20% by time within subject @ fixed 0.50")









#################
# Packages
library(dplyr)
library(ggplot2)
library(Rtsne)

# --- Inputs from your setup (already defined in your message) ---
# GHQ_cat_df <- read.csv("PED_GHQ_categorical_data_forecast.csv")
# columns_GHQ_cat_df <- read.csv("columns_PED_GHQ_categorical_data_forecast.csv")
# get_cols <- function(flag) {
#   columns_GHQ_cat_df %>% dplyr::filter(.data[[flag]] == 1) %>% dplyr::pull(column_name)
# }
# GHQ_cat_outcome_cols         <- get_cols("outcomes")
# GHQ_cat_participant_cols     <- get_cols("participant_id")
# GHQ_cat_time_cols            <- get_cols("time")
# GHQ_cat_fixed_effects_cols   <- get_cols("fixed_effects")
# GHQ_cat_random_effects_cols  <- get_cols("random_effects")
# outcome_col <- GHQ_cat_outcome_cols[1]
# id_col      <- GHQ_cat_participant_cols[1]
# time_col    <- GHQ_cat_time_cols[1]

# Aliases + tidy-eval symbols
id_var   <- id_col
time_var <- time_col
y_var    <- outcome_col
id_sym   <- rlang::sym(id_var)
time_sym <- rlang::sym(time_var)
y_sym    <- rlang::sym(y_var)

set.seed(42)

# If time is a character timestamp, uncomment so ordering is correct:
# GHQ_cat_df[[time_var]] <- as.POSIXct(GHQ_cat_df[[time_var]], tz = "UTC")

# --- 1) Pick 10 subjects; keep ONLY the first 5 measurements per subject
all_ids <- unique(GHQ_cat_df[[id_var]])
if (length(all_ids) == 0) stop("No subjects found in GHQ_cat_df.")

selected_ids <- sample(all_ids, size = min(10, length(all_ids)), replace = FALSE)

df_sub <- GHQ_cat_df %>%
  dplyr::filter(!!id_sym %in% selected_ids) %>%
  dplyr::arrange(!!id_sym, !!time_sym) %>%
  dplyr::group_by(!!id_sym) %>%
  dplyr::slice_head(n = 5) %>%
  dplyr::ungroup()

# --- 2) Build feature matrix from fixed + random effects with robust NA handling
feature_cols <- unique(c(GHQ_cat_fixed_effects_cols, GHQ_cat_random_effects_cols))
feature_cols <- feature_cols[feature_cols %in% names(df_sub)]
if (length(feature_cols) == 0) stop("No fixed/random effect columns present in the selected data.")

X_raw <- df_sub %>% dplyr::select(dplyr::all_of(feature_cols))

# Imputation helpers
impute_numeric_median <- function(x) {
  if (!is.numeric(x)) return(x)
  if (all(is.na(x))) return(x)         # leave all-NA; we'll drop later if needed
  x[is.na(x)] <- stats::median(x, na.rm = TRUE)
  x
}
to_categorical_with_missing <- function(x) {
  if (is.numeric(x)) return(x)
  x_chr <- as.character(x)
  x_chr[is.na(x_chr)] <- "missing"
  factor(x_chr)
}

# Apply imputations (no negation tricks)
X_imp <- X_raw %>%
  dplyr::mutate(dplyr::across(where(is.numeric), impute_numeric_median)) %>%
  dplyr::mutate(dplyr::across(everything(), to_categorical_with_missing))

# Drop all-NA numeric columns if any remain
all_na_cols <- vapply(X_imp, function(col) is.numeric(col) && all(is.na(col)), logical(1))
if (any(all_na_cols)) X_imp <- X_imp[, !all_na_cols, drop = FALSE]
if (ncol(X_imp) == 0) stop("After NA handling, no usable feature columns remain.")

# One-hot encode (no intercept), then drop zero-variance columns
X <- stats::model.matrix(~ . - 1, data = X_imp)
if (!is.matrix(X)) X <- as.matrix(X)
keep <- apply(X, 2, function(col) stats::sd(col, na.rm = TRUE) > 0)
X <- X[, keep, drop = FALSE]
if (ncol(X) == 0) stop("All encoded feature columns have zero variance.")

# Meta
meta <- df_sub %>% dplyr::select(!!id_sym, !!time_sym, !!y_sym)

# y == 1 flag (supports numeric or "1" as char/factor)
yv <- meta[[y_var]]
meta$y_is_one <- if (is.numeric(yv)) (!is.na(yv) & yv == 1) else (!is.na(yv) & as.character(yv) == "1")

# Scale features
X_scaled <- scale(X)

# --- 3) t-SNE
n_obs <- nrow(X_scaled)
if (n_obs < 5) stop(paste0("Too few rows for t-SNE (n=", n_obs, "). Need at least 5."))
perp <- max(2, min(30, floor((n_obs - 1) / 3)))

tsne_out <- Rtsne(
  X_scaled,
  perplexity       = perp,
  check_duplicates = FALSE,
  pca              = TRUE,
  theta            = 0.5,
  max_iter         = 1000,
  verbose          = TRUE
)

emb <- as.data.frame(tsne_out$Y)
names(emb) <- c("TSNE1", "TSNE2")
plot_df <- dplyr::bind_cols(meta, emb)

# Ensure DISCRETE colors for subjects
plot_df[[id_var]] <- factor(plot_df[[id_var]])

# --- 4) Plot: color by SUBJECT (discrete palette); outline points where y == 1
p <- ggplot(plot_df, aes(x = TSNE1, y = TSNE2)) +
  geom_point(aes(color = !!id_sym), size = 2.6, alpha = 0.9) +
  geom_point(
    data  = dplyr::filter(plot_df, y_is_one),
    shape = 21, fill = NA, color = "black",
    stroke = 1.1, size = 3.6
  ) +
  scale_color_brewer(palette = "Set3", na.translate = FALSE) +  # 12 distinct colors (enough for 10 subjects)
  labs(
    title = "t-SNE of Fixed + Random Effects (10 subjects, first 5 measurements each) — GHQ",
    subtitle = paste0("Colored by subject; black outline marks y == 1 (n=", n_obs, ", perplexity=", perp, ")"),
    color = "Subject"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

print(p)

# --- 5) Quick check: rows kept per subject (≤ 5 each)
df_sub %>%
  dplyr::count(!!id_sym, name = "rows_kept") %>%
  dplyr::arrange(!!id_sym) %>%
  print(n = Inf)




