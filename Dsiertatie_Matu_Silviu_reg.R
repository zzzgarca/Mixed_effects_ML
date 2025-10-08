# ================== Packages ==================
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, tibble, purrr, stringr, readr, nlme)

# ================== Setup ==================
working_folder_path <- "/Users/silviumatu/Desktop/Code/R/Disertatie"
setwd(working_folder_path)
options(max.print = 9999, width = 500)
set.seed(1234)

# ================== Data ==================
EXP_reg_df <- read.csv("EXP_regression_data_forecast.csv")
columns_EXP_reg_df <- read.csv("columns_EXP_regression_data_forecast.csv")

get_cols <- function(flag) {
  columns_EXP_reg_df %>% filter(.data[[flag]] == 1) %>% pull(column_name)
}

# Flags
EXP_reg_outcome_cols         <- get_cols("outcomes")
EXP_reg_participant_cols     <- get_cols("participant_id")
EXP_reg_time_cols            <- get_cols("time")
EXP_reg_fixed_effects_cols   <- get_cols("fixed_effects")
EXP_reg_random_effects_cols  <- get_cols("random_effects")

# Columns
outcome_col <- EXP_reg_outcome_cols[1]
id_col      <- EXP_reg_participant_cols[1]
time_col    <- EXP_reg_time_cols[1]

# ================== Basic typing & de-dup ==================
EXP_reg_df[[id_col]] <- as.factor(EXP_reg_df[[id_col]])
if (!is.numeric(EXP_reg_df[[time_col]])) {
  parsed <- suppressWarnings(as.POSIXct(EXP_reg_df[[time_col]], tz = "UTC"))
  EXP_reg_df[[time_col]] <-
    if (all(!is.na(parsed))) as.numeric(parsed) else as.numeric(factor(EXP_reg_df[[time_col]]))
}

# Audit duplicates by (id, time)
dup_summary <- EXP_reg_df %>%
  dplyr::group_by(.data[[id_col]], .data[[time_col]]) %>%
  dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
  dplyr::filter(n > 1)
cat("Duplicate (id, time) rows:", nrow(dup_summary), "\n")
if (nrow(dup_summary) > 0) print(utils::head(dup_summary, 10))

# Remove duplicates: keep first row within each (id, time)
n_before <- nrow(EXP_reg_df)
EXP_reg_df <- EXP_reg_df %>%
  dplyr::arrange(.data[[id_col]], .data[[time_col]]) %>%
  dplyr::group_by(.data[[id_col]], .data[[time_col]]) %>%
  dplyr::slice_head(n = 1) %>%
  dplyr::ungroup()
n_after <- nrow(EXP_reg_df)
cat("Rows removed due to duplicate (id, time): ", n_before - n_after, "\n")

# Keep rows with non-missing outcome/id
EXP_reg_df <- EXP_reg_df %>%
  filter(!is.na(.data[[outcome_col]]), !is.na(.data[[id_col]]))

# ================== Helpers ==================
btick <- function(x) paste0("`", x, "`")

# Regression metrics
compute_metrics_reg <- function(y, yhat) {
  y    <- as.numeric(y)
  yhat <- as.numeric(yhat)
  
  # drop non-finite pairs
  ok <- is.finite(y) & is.finite(yhat)
  y <- y[ok]; yhat <- yhat[ok]
  
  resid <- y - yhat
  sse <- sum(resid^2)
  sst <- sum((y - mean(y))^2)
  
  r <- if (sd(y) > 0 && sd(yhat) > 0) cor(y, yhat) else NA_real_
  
  tibble(
    task_1_MAE      = mean(abs(resid)),
    task_1_RMSE     = sqrt(mean(resid^2)),
    task_1_PearsonR = r,
    task_1_R2       = if (sst > 0) 1 - sse/sst else NA_real_
  )
}


# CI summarizer
mean_ci <- function(x, conf = 0.95) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n == 0) return(c(mean=NA, lwr=NA, upr=NA))
  m <- mean(x); s <- sd(x); tcrit <- qt(1-(1-conf)/2, df=max(1,n-1)); me <- tcrit*s/sqrt(n)
  c(mean=m, lwr=m-me, upr=m+me)
}
summarize_cv <- function(df_metrics, label) {
  df_metrics %>%
    pivot_longer(starts_with("task_1_"), names_to="metric", values_to="value") %>%
    group_by(set, metric) %>%
    summarize(mean=mean(value, na.rm=TRUE),
              lwr=mean_ci(value)["lwr"], upr=mean_ci(value)["upr"], .groups="drop") %>%
    mutate(mean=sprintf("%.4f", mean), lwr=sprintf("%.4f", lwr), upr=sprintf("%.4f", upr),
           split_type=label, summary=paste0(mean, "  (95% CI ", lwr, ", ", upr, ")")) %>%
    select(split_type, set, metric, summary) %>% arrange(split_type, set, metric)
}
format_points <- function(dfrow, label) {
  dfrow %>% pivot_longer(everything(), names_to="metric", values_to="value") %>%
    transmute(split_type=label, metric, summary=sprintf("%.4f", value)) %>%
    arrange(metric)
}

# Drop constants (0-variance numerics or single-level factors/chars)
prune_constant_vars <- function(df, cols) {
  keep <- character(0); drop <- character(0)
  for (nm in cols) {
    v <- df[[nm]]; if (is.null(v)) next
    if (is.numeric(v)) {
      if (sum(!is.na(v)) == 0) { drop <- c(drop, nm); next }
      s <- sd(v, na.rm = TRUE)
      if (is.finite(s) && s > 0) keep <- c(keep, nm) else drop <- c(drop, nm)
    } else {
      u <- unique(v[!is.na(v)])
      if (length(u) > 1) keep <- c(keep, nm) else drop <- c(drop, nm)
    }
  }
  list(keep = unique(keep), drop = unique(drop))
}

# ================== Global constant prune (optional) ==================
pf_fixed  <- prune_constant_vars(EXP_reg_df, EXP_reg_fixed_effects_cols)
fixed_cols <- pf_fixed$keep
if (length(pf_fixed$drop)) message("Dropped global-constant fixed effects: ", paste(pf_fixed$drop, collapse=", "))

pf_random <- prune_constant_vars(EXP_reg_df, EXP_reg_random_effects_cols)
random_cols <- intersect(pf_random$keep, fixed_cols)  # ensure random ⊂ fixed after prune
if (length(setdiff(EXP_reg_random_effects_cols, random_cols))) {
  message("Dropped global-constant random candidates: ",
          paste(setdiff(EXP_reg_random_effects_cols, random_cols), collapse=", "))
}

# ================== Train-only PCA (fit on train, apply to test) ==================
choose_k <- function(pr, thresh = 0.95) {
  if (is.null(pr) || is.null(pr$sdev)) return(0L)
  ev <- pr$sdev^2; which(cumsum(ev / sum(ev)) >= thresh)[1]
}

fit_pca_train_transform <- function(train_df, test_df, fixed_cols, random_cols,
                                    var_exp = 0.95, cap_shared = 5L, nz_thres = 1e-12) {
  fixed_only_cols <- setdiff(fixed_cols, random_cols)
  shared_cols     <- intersect(fixed_cols, random_cols)
  
  keep_numeric_nz <- function(df, cols) {
    cols[vapply(cols, function(c) {
      x <- df[[c]]
      is.numeric(x) && sum(!is.na(x)) >= 2 && stats::sd(x, na.rm = TRUE) > nz_thres
    }, logical(1))]
  }
  impute_mat <- function(mat, means) {
    if (!length(means)) return(mat)
    for (j in seq_along(means)) {
      idx <- is.na(mat[, j]); if (any(idx)) mat[idx, j] <- means[j]
    }
    mat
  }
  score_block <- function(tr_df, te_df, cols, var_exp, cap, prefix) {
    cols <- keep_numeric_nz(tr_df, cols)
    if (!length(cols)) return(list(tr=tibble(), te=tibble(), names=character(0)))
    
    tr_mat <- as.matrix(tr_df[, cols, drop=FALSE])
    te_mat <- as.matrix(te_df[, cols, drop=FALSE])
    
    # Train-mean impute (train params only)
    tr_means <- suppressWarnings(colMeans(tr_mat, na.rm = TRUE))
    tr_mat <- impute_mat(tr_mat, tr_means)
    te_mat <- impute_mat(te_mat, tr_means)
    
    pr <- prcomp(tr_mat, center = TRUE, scale. = TRUE)
    
    k <- max(1L, choose_k(pr, var_exp))
    if (is.finite(cap)) k <- min(k, as.integer(cap))
    
    # Transform both using TRAIN center/scale/rotation
    tr_scaled <- sweep(tr_mat, 2, pr$center, "-"); tr_scaled <- sweep(tr_scaled, 2, pr$scale, "/")
    te_scaled <- sweep(te_mat, 2, pr$center, "-"); te_scaled <- sweep(te_scaled, 2, pr$scale, "/")
    
    PC_tr <- tr_scaled %*% pr$rotation[, 1:k, drop=FALSE]
    PC_te <- te_scaled %*% pr$rotation[, 1:k, drop=FALSE]
    
    nm <- paste0(prefix, seq_len(k))
    list(
      tr = as_tibble(PC_tr) |> `colnames<-`(nm),
      te = as_tibble(PC_te) |> `colnames<-`(nm),
      names = nm
    )
  }
  
  blk_f <- score_block(train_df, test_df, fixed_only_cols, var_exp, Inf,        "PCf_")
  blk_s <- score_block(train_df, test_df, shared_cols,     var_exp, cap_shared, "PCs_")
  
  list(
    train  = bind_cols(train_df, blk_f$tr, blk_s$tr),
    test   = bind_cols(test_df,  blk_f$te, blk_s$te),
    pc_fixed_names  = blk_f$names,
    pc_shared_names = blk_s$names
  )
}

# ================== Model fit/predict (nlme::lme) ==================
fit_lme_and_score <- function(train_df, test_df,
                              outcome_col, id_col, time_col,
                              pc_fixed_names, pc_shared_names,
                              use_AR1 = FALSE,
                              pred_level = 0,               # 0 = population; 1 = include RE
                              method = c("ML","REML")) {   # ML for CV; REML for final
  method <- match.arg(method)
  
  use_cols <- c(outcome_col, id_col, time_col, pc_fixed_names, pc_shared_names)
  tr <- train_df %>% dplyr::select(dplyr::all_of(use_cols)) %>% tidyr::drop_na()
  te <- test_df  %>% dplyr::select(dplyr::all_of(use_cols)) %>% tidyr::drop_na()
  
  tr[[id_col]] <- factor(tr[[id_col]])
  te[[id_col]] <- factor(te[[id_col]], levels = levels(tr[[id_col]]))
  
  rhs <- if (length(c(pc_fixed_names, pc_shared_names)))
    paste(btick(c(pc_fixed_names, pc_shared_names)), collapse = " + ") else "1"
  fixed_form <- as.formula(paste(btick(outcome_col), "~", rhs))
  
  # Diagonal random effects (intercept + shared PCs)
  rand_list <- setNames(
    list(nlme::pdDiag(as.formula(paste("~", paste(c("1", pc_shared_names), collapse = " + "))))),
    id_col
  )
  
  corr <- NULL
  if (use_AR1) {
    corr <- nlme::corAR1(form = as.formula(paste("~", btick(time_col), "|", btick(id_col))))
  }
  
  ctrl <- nlme::lmeControl(msMaxIter = 200, msMaxEval = 200, opt = "nlminb")
  fit <- nlme::lme(
    fixed  = fixed_form,
    random = rand_list,
    data   = tr,
    method = method,
    correlation = corr,
    control = ctrl
  )
  
  yhat_tr <- as.numeric(predict(fit, newdata = tr, level = pred_level))
  yhat_te <- as.numeric(predict(fit, newdata = te, level = pred_level))
  
  list(
    train_metrics = compute_metrics_reg(tr[[outcome_col]], yhat_tr),
    test_metrics  = compute_metrics_reg(te[[outcome_col]], yhat_te),
    model = fit
  )
}

# ================== Splitters ==================
# A) Subject-based K-fold (hold out whole participants)
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

# B) Time-based K-fold per subject (contiguous chunks)
time_kfold_splits_by_subject <- function(df, id_col, time_col, k = 5, min_points = 4) {
  df2 <- df %>% arrange(.data[[id_col]], .data[[time_col]]) %>% mutate(.row_id = dplyr::row_number())
  by_id <- df2 %>% group_by(.data[[id_col]]) %>% group_split()
  folds <- replicate(k, list(train_idx=integer(0), test_idx=integer(0)), simplify=FALSE)
  for (g in by_id) {
    n_i <- nrow(g)
    if (n_i >= max(min_points, k)) {
      cuts <- floor(seq(0, n_i, length.out = k + 1))
      for (i in seq_len(k)) {
        idx <- if (cuts[i] < cuts[i+1]) seq.int(cuts[i]+1, cuts[i+1]) else integer(0)
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
    train_df = df %>% filter(.data[[id_col]] %in% train_ids),
    test_df  = df %>% filter(!(.data[[id_col]] %in% train_ids))
  )
}
split_time_lastX_by_subject <- function(df, id_col, time_col, min_points = 4, test_frac = 0.30) {
  df2 <- df %>% arrange(.data[[id_col]], .data[[time_col]]) %>% mutate(.row_id = dplyr::row_number())
  counts <- df2 %>% count(.data[[id_col]], name = "n_i")
  eligible_ids <- counts %>% filter(n_i >= min_points) %>% pull(.data[[id_col]])
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
  list(
    train_df = df2 %>% filter(.row_id %in% idx_train) %>% select(-.row_id),
    test_df  = df2 %>% filter(.row_id %in% idx_test)  %>% select(-.row_id),
    eligible_ids = eligible_ids
  )
}

# ================== Fold runner (per-fold PCA; selectable AR1/level/method) ==================
run_kfold <- function(df, folds, label,
                      fixed_cols, random_cols,
                      outcome_col, id_col, time_col,
                      var_exp = 0.95, cap_shared = 5L,
                      use_AR1 = FALSE, pred_level = 0, method = "ML") {
  out <- vector("list", length(folds))
  for (i in seq_along(folds)) {
    tr <- df[folds[[i]]$train_idx, , drop = FALSE]
    te <- df[folds[[i]]$test_idx,  , drop = FALSE]
    if (!nrow(te) || !nrow(tr)) next
    
    pca <- fit_pca_train_transform(tr, te, fixed_cols, random_cols,
                                   var_exp = var_exp, cap_shared = cap_shared)
    
    res <- fit_lme_and_score(
      pca$train, pca$test,
      outcome_col, id_col, time_col,
      pca$pc_fixed_names, pca$pc_shared_names,
      use_AR1 = use_AR1, pred_level = pred_level, method = method
    )
    
    out[[i]] <- bind_rows(
      res$train_metrics %>% mutate(set = "train"),
      res$test_metrics  %>% mutate(set = "test")
    ) %>% mutate(fold = i, split_type = label)
  }
  bind_rows(out)
}

# ================== Run K-fold CV (NO leakage; ML for CV) ==================
k <- 5  # change as needed

# Subject-based CV (NEW subjects in test): population predictions, no AR(1)
folds_subject <- subject_kfold_splits(EXP_reg_df, id_col = id_col, k = k, seed = 42)
cv_subject <- run_kfold(
  EXP_reg_df, folds_subject, label = "subject_kfold",
  fixed_cols = fixed_cols, random_cols = random_cols,
  outcome_col = outcome_col, id_col = id_col, time_col = time_col,
  var_exp = 0.95, cap_shared = 5L,
  use_AR1 = FALSE, pred_level = 0, method = "ML"
)
summary_subject <- summarize_cv(cv_subject, label = "subject_kfold")

# Time-based CV (same subjects across time): AR(1) + include RE in predictions
folds_time <- time_kfold_splits_by_subject(EXP_reg_df, id_col = id_col, time_col = time_col, k = k, min_points = 2)
cv_time <- run_kfold(
  EXP_reg_df, folds_time, label = "time_kfold",
  fixed_cols = fixed_cols, random_cols = random_cols,
  outcome_col = outcome_col, id_col = id_col, time_col = time_col,
  var_exp = 0.95, cap_shared = 5L,
  use_AR1 = TRUE, pred_level = 1, method = "ML"
)
summary_time <- summarize_cv(cv_time, label = "time_kfold")

# Print CV summaries
bind_rows(summary_subject, summary_time) %>% arrange(split_type, set, metric) %>% print(n = Inf)

# ================== Final single-split fits (NO CV; REML for final) ==================

# A) 80/20 by participants (NEW subjects in test): population predictions, no AR(1)
ss_part <- single_split_by_participants_80_20(EXP_reg_df, id_col, seed = 123)
pca_part <- fit_pca_train_transform(ss_part$train_df, ss_part$test_df, fixed_cols, random_cols,
                                    var_exp = 0.95, cap_shared = 5L)
final_part <- fit_lme_and_score(
  pca_part$train, pca_part$test,
  outcome_col, id_col, time_col,
  pca_part$pc_fixed_names, pca_part$pc_shared_names,
  use_AR1 = FALSE, pred_level = 0, method = "REML"
)
cat("\n--- Final 80/20 by participants (population-level predictions) ---\n")
bind_rows(
  format_points(final_part$train_metrics, "subject_80_20_train"),
  format_points(final_part$test_metrics,  "subject_80_20_test")
) %>% arrange(split_type, metric) %>% print(n = Inf)

# B) Last 30% by time within subject (≥4 points): AR(1) + include RE
ss_time <- split_time_lastX_by_subject(EXP_reg_df, id_col, time_col, min_points = 4, test_frac = 0.30)
pca_time <- fit_pca_train_transform(ss_time$train_df, ss_time$test_df, fixed_cols, random_cols,
                                    var_exp = 0.95, cap_shared = 5L)
final_time <- fit_lme_and_score(
  pca_time$train, pca_time$test,
  outcome_col, id_col, time_col,
  pca_time$pc_fixed_names, pca_time$pc_shared_names,
  use_AR1 = TRUE, pred_level = 1, method = "REML"
)
cat("\n--- Final time split: last 30% per eligible subject (with RE & AR1) ---\n")
bind_rows(
  format_points(final_time$train_metrics, "time_last30_by_subject_train"),
  format_points(final_time$test_metrics,  "time_last30_by_subject_test")
) %>% arrange(split_type, metric) %>% print(n = Inf)




# Packages
library(dplyr)
library(ggplot2)
library(Rtsne)

# --- Inputs from your setup ---
# EXP_reg_df <- read.csv("EXP_regression_data_forecast.csv")
# columns_EXP_reg_df <- read.csv("columns_EXP_regression_data_forecast.csv")
# get_cols <- function(flag) {
#   columns_EXP_reg_df %>% dplyr::filter(.data[[flag]] == 1) %>% dplyr::pull(column_name)
# }
# EXP_reg_outcome_cols         <- get_cols("outcomes")
# EXP_reg_participant_cols     <- get_cols("participant_id")
# EXP_reg_time_cols            <- get_cols("time")
# EXP_reg_fixed_effects_cols   <- get_cols("fixed_effects")
# EXP_reg_random_effects_cols  <- get_cols("random_effects")
# outcome_col <- EXP_reg_outcome_cols[1]
# id_col      <- EXP_reg_participant_cols[1]
# time_col    <- EXP_reg_time_cols[1]

# Aliases + tidy-eval symbols
id_var   <- id_col
time_var <- time_col
y_var    <- outcome_col
id_sym   <- rlang::sym(id_var)
time_sym <- rlang::sym(time_var)
y_sym    <- rlang::sym(y_var)

set.seed(42)

# If time is character, uncomment for correct ordering:
# EXP_reg_df[[time_var]] <- as.POSIXct(EXP_reg_df[[time_var]], tz = "UTC")

# --- 1) Pick 10 subjects; keep ONLY the first 5 measurements per subject
all_ids <- unique(EXP_reg_df[[id_var]])
if (length(all_ids) == 0) stop("No subjects found in EXP_reg_df.")

selected_ids <- sample(all_ids, size = min(10, length(all_ids)), replace = FALSE)

df_sub <- EXP_reg_df %>%
  dplyr::filter(!!id_sym %in% selected_ids) %>%
  dplyr::arrange(!!id_sym, !!time_sym) %>%
  dplyr::group_by(!!id_sym) %>%
  dplyr::slice_head(n = 5) %>%
  dplyr::ungroup()

# --- 2) Feature matrix from fixed + random effects with robust NA handling
feature_cols <- unique(c(EXP_reg_fixed_effects_cols, EXP_reg_random_effects_cols))
feature_cols <- feature_cols[feature_cols %in% names(df_sub)]
if (length(feature_cols) == 0) stop("No fixed/random effect columns present in the selected data.")

X_raw <- df_sub %>% dplyr::select(dplyr::all_of(feature_cols))

# Imputation helpers
impute_numeric_median <- function(x) {
  if (!is.numeric(x)) return(x)
  if (all(is.na(x))) return(x)
  x[is.na(x)] <- stats::median(x, na.rm = TRUE)
  x
}
to_categorical_with_missing <- function(x) {
  if (is.numeric(x)) return(x)
  x_chr <- as.character(x)
  x_chr[is.na(x_chr)] <- "missing"
  factor(x_chr)
}

# Apply imputations
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

# --- 4) Plot: color by SUBJECT (no y==1 margin)
p <- ggplot(plot_df, aes(x = TSNE1, y = TSNE2)) +
  geom_point(aes(color = !!id_sym), size = 2.6, alpha = 0.9) +
  scale_color_brewer(palette = "Set3", na.translate = FALSE) +
  labs(
    title = "t-SNE of Fixed + Random Effects (10 subjects, first 5 measurements each) — EXP (regression)",
    subtitle = paste0("Colored by subject (n=", n_obs, ", perplexity=", perp, ")"),
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







library(dplyr)
library(tidyr)
library(ggplot2)
library(nlme)
library(tibble)
library(scales)

# --- pick the fitted lme model you already created ---
model_obj <- if (exists("final_time") && inherits(final_time$model, "lme")) {
  final_time$model
} else if (exists("final_part") && inherits(final_part$model, "lme")) {
  final_part$model
} else {
  stop("No fitted lme model found (looked for final_time$model or final_part$model).")
}

# --- extract subject-level random effects (BLUPs) ---
# For a single grouping factor, nlme::ranef() returns a data.frame:
# columns = random effects (e.g., (Intercept), PCs_1, PCs_2, ...)
re_df <- nlme::ranef(model_obj) %>% as.data.frame()

# Defensive check: drop any all-NA columns before correlation
non_all_na <- vapply(re_df, function(z) !all(is.na(z)), logical(1))
re_df <- re_df[, non_all_na, drop = FALSE]
if (ncol(re_df) < 2) stop("Fewer than 2 random-effect columns available to correlate.")

# --- correlation matrix ---
corr_mat <- stats::cor(re_df, use = "pairwise.complete.obs")

# Tidy for plotting
corr_long <- corr_mat %>%
  as.data.frame() %>%
  rownames_to_column("Var1") %>%
  pivot_longer(-Var1, names_to = "Var2", values_to = "r") %>%
  mutate(
    Var1 = factor(Var1, levels = colnames(corr_mat)),
    Var2 = factor(Var2, levels = colnames(corr_mat))
  )

# --- heatmap plot ---
ggplot(corr_long, aes(x = Var1, y = Var2, fill = r)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3) +
  scale_fill_gradient2(limits = c(-1, 1), oob = squish, name = "Correlation") +
  coord_fixed() +
  labs(
    title = "Correlation of Random Effects (subject-level BLUPs)",
    x = NULL, y = NULL,
    subtitle = "Computed from posterior modes of random effects per subject"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))




library(dplyr)
library(tidyr)
library(ggplot2)

# --- 1) Find 10 participants with >= 10 non-missing outcome measurements
eligible_ids <- EXP_reg_df %>%
  dplyr::filter(!is.na(.data[[outcome_col]])) %>%
  dplyr::group_by(.data[[id_col]]) %>%
  dplyr::summarise(n_meas = dplyr::n(), .groups = "drop") %>%
  dplyr::filter(n_meas >= 10) %>%
  dplyr::arrange(.data[[id_col]]) %>%
  dplyr::pull(.data[[id_col]])

n_pick <- min(10, length(eligible_ids))
picked_ids <- eligible_ids[seq_len(n_pick)]
if (n_pick < 10) message("Only ", n_pick, " participants met the >=10 criterion.")

# --- 2) Extract first 10 available measurements and index them 1..10
scores_long <- EXP_reg_df %>%
  dplyr::filter(.data[[id_col]] %in% picked_ids, !is.na(.data[[outcome_col]])) %>%
  dplyr::arrange(.data[[id_col]], .data[[time_col]]) %>%
  dplyr::group_by(.data[[id_col]]) %>%
  dplyr::slice_head(n = 10) %>%
  dplyr::mutate(meas_idx = dplyr::row_number()) %>%
  dplyr::ungroup()

# --- 3) Simple per-participant fit: y ~ meas_idx (uses only these 10 points)
fmla <- as.formula(paste(outcome_col, "~ meas_idx"))

pred_by_case <- scores_long %>%
  dplyr::group_by(.data[[id_col]]) %>%
  dplyr::group_modify(function(.x, .y) {
    fit <- lm(fmla, data = .x)
    .x$y_hat <- as.numeric(predict(fit, newdata = .x))
    .x
  }) %>%
  dplyr::ungroup()

# --- 4) Long format for plotting: Observed vs Fitted
plot_df <- pred_by_case %>%
  dplyr::transmute(
    id = .data[[id_col]],
    meas_idx,
    observed = .data[[outcome_col]],
    fitted   = y_hat
  ) %>%
  tidyr::pivot_longer(c(observed, fitted), names_to = "series", values_to = "value") %>%
  dplyr::mutate(series = dplyr::recode(series, observed = "Observed", fitted = "Fitted"))

# --- 5) Plot: same color per participant, different linetype for observed vs fitted
ggplot(
  plot_df,
  aes(x = meas_idx, y = value,
      color = id,
      linetype = series,
      group = interaction(id, series))
) +
  geom_line(size = 1) +
  geom_point(size = 2, alpha = 0.8) +
  scale_x_continuous(breaks = 1:10, limits = c(1, 10)) +
  scale_linetype_manual(values = c("Observed" = "solid", "Fitted" = "dashed")) +
  labs(
    title = "Observed vs Fitted (per-participant linear fit on first 10 measurements)",
    subtitle = paste0("Participants: ", n_pick, " with ≥10 measurements; model: ", outcome_col, " ~ meas_idx"),
    x = "Measurement index (1..10 within subject)",
    y = outcome_col,
    color = "Participant",
    linetype = "Series"
  ) +
  theme_minimal() +
  theme(legend.position = "right")




