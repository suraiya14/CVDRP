# ============================
# Feature Reduction Script (|r| < 0.90; Chi-square p > 0.001)
# ============================

# --- Settings ---

input_csv <- "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\revised_data\\validate_train_data.csv"
pearson_keep_threshold <- 0.90     # keep pairs with |r| < 0.90
chisq_keep_alpha <- 0.001          # keep pairs with p > 0.001 (declare conflict when p <= 0.001)
outcome_col <- "CVD_risk"

declared_categorical <- c(
  "SMQ020","PAD790U","DIQ010","RIAGENDR","RIDRETH3",
  "DMDEDUC2","BPQ101D","BPQ020","BPQ080","RXQ033"
)

# --- Load data ---
df <- read.csv(input_csv, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
if (outcome_col %in% names(df)) df[[outcome_col]] <- NULL

categorical_cols <- intersect(declared_categorical, colnames(df))
is_numeric <- vapply(df, is.numeric, logical(1))
continuous_cols <- setdiff(names(df)[is_numeric], categorical_cols)
for (cn in categorical_cols) if (!is.factor(df[[cn]])) df[[cn]] <- as.factor(df[[cn]])

# --- Continuous reduction: drop until all |r| < threshold ---
reduce_continuous_by_corr <- function(dat, cols, keep_thr = 0.90) {
  cols <- intersect(cols, colnames(dat))
  if (length(cols) <= 1L) return(cols)
  M <- stats::cor(dat[, cols, drop = FALSE], use = "pairwise.complete.obs")
  M[is.na(M)] <- 0
  to_drop <- character(0)
  
  repeat {
    A <- abs(M); diag(A) <- 0
    max_val <- max(A)
    # stop when the largest correlation is already under the keep threshold
    if (!is.finite(max_val) || max_val < keep_thr) break
    
    # get the pair with the max correlation (>= keep_thr)
    idx <- which(A == max_val, arr.ind = TRUE)[1, ]
    i <- rownames(A)[idx[1]]; j <- colnames(A)[idx[2]]
    
    # drop the one more "redundant" (higher mean |cor| to others)
    mean_i <- mean(A[i, setdiff(colnames(A), i)], na.rm = TRUE)
    mean_j <- mean(A[j, setdiff(colnames(A), j)], na.rm = TRUE)
    drop_col <- if (mean_i >= mean_j) i else j
    to_drop <- c(to_drop, drop_col)
    
    keep <- setdiff(rownames(M), drop_col)
    if (length(keep) <= 1L) {
      M <- as.matrix(M[keep, keep, drop = FALSE]); break
    } else {
      M <- stats::cor(dat[, keep, drop = FALSE], use = "pairwise.complete.obs")
      M[is.na(M)] <- 0
    }
  }
  setdiff(cols, unique(to_drop))
}

reduced_continuous <- reduce_continuous_by_corr(df, continuous_cols, keep_thr = pearson_keep_threshold)

# --- Categorical reduction: drop until all pairwise p > alpha ---
safe_chisq_p <- function(x, y) {
  tb <- table(x, y, useNA = "no")
  if (nrow(tb) < 2 || ncol(tb) < 2 || all(tb == 0)) return(1)
  pval <- tryCatch({ suppressWarnings(stats::chisq.test(tb)$p.value) }, error = function(e) 1)
  if (is.na(pval)) 1 else pval
}

reduce_categorical_by_chisq <- function(dat, cols, keep_alpha = 1e-3) {
  cols <- intersect(cols, colnames(dat))
  if (length(cols) <= 1L) return(cols)
  for (cn in cols) if (!is.factor(dat[[cn]])) dat[[cn]] <- as.factor(dat[[cn]])
  
  current <- cols
  repeat {
    if (length(current) <= 1L) break
    pairs <- combn(current, 2, simplify = FALSE)
    if (length(pairs) == 0) break
    
    pvals <- vapply(pairs, function(pr) safe_chisq_p(dat[[pr[1]]], dat[[pr[2]]]), numeric(1))
    
    # declare a "conflict" when p <= alpha (i.e., significantly dependent)
    conflicted <- which(pvals <= keep_alpha)
    if (length(conflicted) == 0) break
    
    # score variables by overall dependency (-log10 p across all pairs)
    dep_scores <- setNames(numeric(length(current)), current)
    for (k in seq_along(pairs)) {
      pv <- pvals[k]; score <- -log10(pv)
      if (is.finite(score)) {
        pr <- pairs[[k]]
        dep_scores[pr[1]] <- dep_scores[pr[1]] + score
        dep_scores[pr[2]] <- dep_scores[pr[2]] + score
      }
    }
    vars_in_conflict <- unique(unlist(pairs[conflicted]))
    drop_var <- names(sort(dep_scores[vars_in_conflict], decreasing = TRUE))[1]
    current <- setdiff(current, drop_var)
  }
  current
}

reduced_categorical <- reduce_categorical_by_chisq(df, categorical_cols, keep_alpha = chisq_keep_alpha)

# --- Combined output ---
reduced_all <- unique(c(reduced_continuous, reduced_categorical))

cat("\n--- Reduced continuous columns (all pairwise |r| <", pearson_keep_threshold, ") ---\n")
print(reduced_continuous)

cat("\n--- Reduced categorical columns (all pairwise chi-square p >", chisq_keep_alpha, ") ---\n")
print(reduced_categorical)

cat("\n--- Combined reduced columns ---\n")
print(reduced_all)

# Optionally write to CSV files with one column name per row
write.csv(data.frame(column = reduced_continuous), "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\pearsonAndChi-squared\\reduced_continuous_cols.csv", row.names = FALSE)
write.csv(data.frame(column = reduced_categorical), "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\pearsonAndChi-squared\\reduced_categorical_cols.csv", row.names = FALSE)
write.csv(data.frame(column = reduced_all), "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\pearsonAndChi-squared\\reduced_all_cols.csv", row.names = FALSE)
