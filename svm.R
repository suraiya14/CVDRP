# -------------------------------
# Tuned SVM (RBF) — 10-fold CV
# Evaluation metrics, confusion matrices
# SHAP on ALL training rows + ALL features (fastshap)
# SHAP bar (top 25 + top 10), beeswarm subsets, renamed features, categorical SHAP boxplots
# -------------------------------

suppressPackageStartupMessages({
  library(e1071)      # svm, tune.svm
  library(caret)      # confusionMatrix, nearZeroVar
  library(pROC)       # roc, auc
  library(ggplot2)    # plotting
  library(data.table) # data ops
  library(fastshap)   # approximate SHAP
  # library(doParallel) # optional for parallel SHAP
})

set.seed(123)

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------

df <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin5\\data50\\selected_training_merged_file.csv",
               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
df_test <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin5\\data50\\selected_validation_merged_file.csv",
                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

# -------------------------------------------------
# 2. Helper functions
# -------------------------------------------------
prepare_target <- function(x) {
  # Convert to factor("0","1")
  if (is.factor(x) || is.character(x)) {
    x <- trimws(as.character(x))
    x <- ifelse(x %in% c("Yes","1","TRUE","T"), "1",
                ifelse(x %in% c("No","0","FALSE","F"),  "0", x))
  } else if (is.logical(x)) {
    x <- ifelse(x, "1", "0")
  } else {
    x <- ifelse(as.numeric(x) >= 1, "1", "0")
  }
  factor(x, levels = c("0","1"))
}

char_to_factor <- function(d) {
  char_cols <- vapply(d, is.character, logical(1))
  d[char_cols] <- lapply(d[char_cols], factor)
  d
}

# -------------------------------------------------
# 3. Prepare train/test data
# -------------------------------------------------
stopifnot("CVD_risk" %in% names(df), "CVD_risk" %in% names(df_test))

df$CVD_risk      <- prepare_target(df$CVD_risk)
df_test$CVD_risk <- prepare_target(df_test$CVD_risk)

df      <- char_to_factor(df)
df_test <- char_to_factor(df_test)

pred_train   <- setdiff(names(df), "CVD_risk")
pred_test    <- setdiff(names(df_test), "CVD_risk")
common_pred  <- intersect(pred_train, pred_test)
if (length(common_pred) == 0) stop("No common predictors between train and test.")

df_sub      <- df[,      c("CVD_risk", common_pred), drop = FALSE]
df_test_sub <- df_test[, c("CVD_risk", common_pred), drop = FALSE]

# Drop near-zero-variance predictors (based on training only)
nzv_idx <- nearZeroVar(df_sub[, common_pred, drop = FALSE], saveMetrics = FALSE)
if (length(nzv_idx) > 0) {
  drop_vars    <- common_pred[nzv_idx]
  df_sub       <- df_sub[,      setdiff(names(df_sub), drop_vars), drop = FALSE]
  df_test_sub  <- df_test_sub[, setdiff(names(df_test_sub), drop_vars), drop = FALSE]
  message("Dropped NZV predictors: ", paste(drop_vars, collapse = ", "))
}
final_pred <- setdiff(names(df_sub), "CVD_risk")

# Clean Inf/NaN for numeric columns in train/test
num_cols <- names(df_sub)[sapply(df_sub, is.numeric)]
for (cc in num_cols) {
  v <- df_sub[[cc]]
  v[!is.finite(v)] <- median(v[is.finite(v)], na.rm = TRUE)
  df_sub[[cc]] <- v
  
  v2 <- df_test_sub[[cc]]
  v2[!is.finite(v2)] <- median(v2[is.finite(v2)], na.rm = TRUE)
  df_test_sub[[cc]] <- v2
}

# -------------------------------------------------
# 4. Tune SVM (10-fold CV, original grids unchanged)
# -------------------------------------------------
cost_grid  <- 2 ^ (-1:3)   # 0.5,1,2,4,8
gamma_grid <- 2 ^ (-5:0)   # 1/32 ... 1

set.seed(123)
svm_tuned <- tune.svm(
  CVD_risk ~ .,
  data        = df_sub,
  kernel      = "radial",
  type        = "C-classification",
  scale       = TRUE,
  probability = TRUE,
  cost        = cost_grid,
  gamma       = gamma_grid,
  tunecontrol = tune.control(cross = 10)  # 10-fold CV (do not change)
)

cat("\n=== SVM tuning results (10-fold CV) ===\n")
print(svm_tuned$best.parameters)

svm_fit <- svm_tuned$best.model

# -------------------------------------------------
# 5. External test evaluation
# -------------------------------------------------
svm_pred <- predict(svm_fit, newdata = df_test_sub, probability = TRUE)
svm_probs_mat <- attr(svm_pred, "probabilities")
if (is.null(svm_probs_mat)) {
  stop("SVM probabilities not available even after tuning.")
}

pos_col <- which(colnames(svm_probs_mat) == "1")
if (length(pos_col) != 1) {
  tl <- levels(df_test_sub$CVD_risk)
  pos_col <- which(colnames(svm_probs_mat) == tl[length(tl)])
  if (length(pos_col) != 1) stop("Couldn't identify positive-class probability column.")
}
probs <- as.numeric(svm_probs_mat[, pos_col])

pred_cls <- factor(ifelse(probs >= 0.5, "1", "0"), levels = c("0","1"))
truth    <- df_test_sub$CVD_risk

cm <- confusionMatrix(pred_cls, truth, positive = "1", mode = "everything")
print(cm)

roc_obj  <- roc(response = truth, predictor = probs, levels = c("0","1"), direction = "<")
auc_val  <- auc(roc_obj)
cat(sprintf("SVM AUC (test): %.4f\n", as.numeric(auc_val)))

tp <- as.numeric(cm$table["1","1"])
tn <- as.numeric(cm$table["0","0"])
fp <- as.numeric(cm$table["1","0"])
fn <- as.numeric(cm$table["0","1"])
mcc <- (tp*tn - fp*fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
cat(sprintf("MCC: %.4f\n", mcc))

accuracy_val <- sum(pred_cls == truth) / length(truth)
cat(sprintf("Accuracy: %.4f\n", accuracy_val))

plot(roc_obj, main = sprintf("ROC — Tuned SVM (RBF) (AUC = %.3f)", auc_val))

# -------------------------------------------------
# 6. Confusion matrix visualizations
# -------------------------------------------------
draw_confusion_matrix <- function(cm,
                                  neg_label = "0",
                                  pos_label = "1") {
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table[pos_label, neg_label])
  FN <- as.numeric(cm$table[neg_label, pos_label])
  TP <- as.numeric(cm$table[pos_label, pos_label])
  res <- c(TN, FP, FN, TP)
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  
  text(250, 405, "Reference",  cex=1.3, font=2)
  text(175, 375, "Prediction", cex=1.3, srt=90, font=2)
  
  text(250, 375, neg_label, cex=1.3, font=2)
  text(250, 350, pos_label, cex=1.3, font=2)
  text(300, 400, neg_label, cex=1.3, font=2)
  text(400, 400, pos_label, cex=1.3, font=2)
  
  text(300, 375, res[1], cex=1.6, font=2)  # TN
  text(400, 375, res[2], cex=1.6, font=2)  # FP
  text(300, 350, res[3], cex=1.6, font=2)  # FN
  text(400, 350, res[4], cex=1.6, font=2)  # TP
}

draw_confusion_matrix2 <- function(cm,
                                   neg_label = "0",
                                   pos_label = "1") {
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table[pos_label, neg_label])
  FN <- as.numeric(cm$table[neg_label, pos_label])
  TP <- as.numeric(cm$table[pos_label, pos_label])
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  plot(c(123, 345), c(300, 452), type = "n",
       xlab="", ylab="", xaxt='n', yaxt='n')
  
  rect(150, 430, 240, 370, col='#AF97D0'); text(195, 435, neg_label, cex=1.2)
  rect(250, 430, 340, 370, col='#A7AD50'); text(295, 435,  pos_label, cex=1.2)
  
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual',    cex=1.3, font=2)
  
  rect(150, 305, 240, 365, col='#A7AD50')
  rect(250, 305, 340, 365, col='#AF97D0')
  
  text(140, 400, neg_label, cex=1.2, srt=90)
  text(140, 335,  pos_label, cex=1.2, srt=90)
  
  text(195, 400, TN, cex=1.6, font=2)  # TN
  text(195, 335, FP, cex=1.6, font=2)  # FP
  text(295, 400, FN, cex=1.6, font=2)  # FN
  text(295, 335, TP, cex=1.6, font=2)  # TP
}

draw_confusion_matrix(cm, neg_label = "0", pos_label = "1")
draw_confusion_matrix2(cm, neg_label = "0", pos_label = "1")

# -------------------------------------------------
# 7. SHAP on ALL train rows + ALL features (fastshap)
#    Uses training data ONLY (not test)
# -------------------------------------------------
train_X_all <- df_sub[, final_pred, drop = FALSE]
train_y_all <- df_sub$CVD_risk

pred_fun_fast <- function(object, newdata) {
  pp <- predict(object, newdata = newdata, probability = TRUE)
  pm <- attr(pp, "probabilities")
  pc <- which(colnames(pm) == "1")
  if (length(pc) != 1) {
    lab <- levels(train_y_all)
    pc <- which(colnames(pm) == lab[length(lab)])
  }
  as.numeric(pm[, pc])
}

# OPTIONAL: parallel speedup (commented to match your style)
# library(doParallel)
# cl <- makeCluster(parallel::detectCores() - 1)
# registerDoParallel(cl)

set.seed(123)
shap_matrix <- fastshap::explain(
  object       = svm_fit,
  X            = train_X_all,
  pred_wrapper = pred_fun_fast,
  nsim         = 50,   # higher = smoother SHAP, slower runtime
  adjust       = TRUE
)

# stopCluster(cl) # if you enabled parallel

# -------------------------------------------------
# 8. Long format SHAP + feature values
# -------------------------------------------------
shap_dt <- as.data.table(shap_matrix)
shap_dt[, row_id := seq_len(nrow(shap_dt))]

shap_long <- melt(
  shap_dt,
  id.vars = "row_id",
  variable.name = "variable",
  value.name = "shap_val"
)

feat_vals_dt <- as.data.table(train_X_all)
feat_vals_dt[, row_id := seq_len(nrow(feat_vals_dt))]
feat_long <- melt(
  feat_vals_dt,
  id.vars = "row_id",
  variable.name = "variable",
  value.name = "feat_val"
)

shap_long <- merge(
  shap_long,
  feat_long,
  by = c("row_id","variable"),
  all.x = TRUE
)

# -------------------------------------------------
# 9. Global SHAP bar plots
#    (A) Top ~25
#    (B) Top 10
#    Axes: Feature on vertical, Mean |SHAP| on horizontal
# -------------------------------------------------
shap_bar_data <- shap_long[, .(
  Mean_SHAP = mean(abs(shap_val), na.rm = TRUE)
), by = variable][order(Mean_SHAP, decreasing = TRUE)]

# --- (A) Top ~25 ---
top_n <- min(25, nrow(shap_bar_data))
shap_bar_top <- shap_bar_data[1:top_n]

gg_shap_bar <- ggplot(shap_bar_top,
                      aes(x = reorder(variable, Mean_SHAP),
                          y = Mean_SHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Feature",
       y = "Mean |SHAP value|") +
  theme_minimal() +
  theme(
    axis.title = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.background = element_blank()
  )
print(gg_shap_bar)

# --- (B) Top 10 ---
top_10 <- min(10, nrow(shap_bar_data))
shap_bar_top10 <- shap_bar_data[1:top_10]

gg_shap_bar_top10 <- ggplot(shap_bar_top10,
                            aes(x = reorder(variable, Mean_SHAP),
                                y = Mean_SHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Feature",
       y = "Mean |SHAP value|") +
  theme_minimal() +
  theme(
    axis.title = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.background = element_blank()
  )
print(gg_shap_bar_top10)

# -------------------------------------------------
# 10. Helper for beeswarm-style SHAP plots
#     - vertical 0 line
#     - color by feature value (Low→High legend)
#     - y tick labels "FEATURE  mean|SHAP|"
# -------------------------------------------------
make_shap_beeswarm <- function(shap_df, feat_order_labels) {
  # pretty y labels from feat_order_labels lookup
  shap_df$variable_pretty <- feat_order_labels[as.character(shap_df$variable)]
  
  # reverse order so most important ends up on top
  shap_df$variable_pretty <- factor(
    shap_df$variable_pretty,
    levels = rev(unique(feat_order_labels))
  )
  
  shap_df[, feat_val_scaled := NA_real_]
  
  for (v in unique(shap_df$variable)) {
    idx <- shap_df$variable == v
    fv  <- shap_df$feat_val[idx]
    
    # robust numeric for color scaling
    if (is.factor(fv)) {
      suppressWarnings(fv_num <- as.numeric(as.character(fv)))
      if (any(is.na(fv_num))) {
        fv_num <- as.numeric(fv)
      }
    } else if (is.character(fv)) {
      suppressWarnings(fv_num <- as.numeric(fv))
      if (any(is.na(fv_num))) {
        fv_num <- as.numeric(factor(fv))
      }
    } else {
      fv_num <- as.numeric(fv)
    }
    
    rng <- range(fv_num, na.rm = TRUE)
    rng_diff <- diff(rng)
    
    if (!is.finite(rng_diff) || rng_diff == 0) {
      shap_df$feat_val_scaled[idx] <- 0.5
    } else {
      shap_df$feat_val_scaled[idx] <- (fv_num - rng[1]) / rng_diff
    }
  }
  
  ggplot(
    shap_df,
    aes(x = shap_val,
        y = variable_pretty,
        color = 1 - feat_val_scaled)  # ONLY CHANGE: reverse mapping
  ) +
    geom_vline(xintercept = 0, color = "black") +
    geom_point(
      alpha = 0.7,
      size  = 1.5,
      position = position_jitter(height = 0.2, width = 0)
    ) +
    scale_color_gradient(
      name   = "Feature value",
      limits = c(0,1),
      breaks = c(1,0),         # ONLY CHANGE: keep labels but flip breaks
      labels = c("Low", "High")
    ) +
    labs(
      x = "SHAP value (impact on model output)",
      y = NULL,
      color = "Feature value"
    ) +
    theme_minimal() +
    theme(
      axis.title.y = element_text(face = "bold"),
      panel.grid.major.y = element_blank(),
      panel.grid.minor.y = element_blank()
    )
}

# -------------------------------------------------
# 12. Beeswarm with renamed axes (Suraiya=RIDAGEYR, Kayes=LBXTC)
# -------------------------------------------------
shap_ren <- copy(shap_long)
shap_ren[variable == "RIDAGEYR", variable := "Suraiya"]
shap_ren[variable == "LBXTC",    variable := "Kayes"]

rename_keep <- c("Suraiya", "Kayes")
rename_keep <- rename_keep[rename_keep %in% shap_ren$variable]

if (length(rename_keep) > 0) {
  shap_ren2 <- shap_ren[variable %in% rename_keep]
  
  mean_abs2 <- shap_ren2[, .(
    MeanAbsSHAP = mean(abs(shap_val), na.rm = TRUE)
  ), by = variable]
  
  mean_abs2[, label := paste0(
    variable, "  ",
    formatC(MeanAbsSHAP, digits = 3, format = "f")
  )]
  
  feat_order_labels2 <- setNames(mean_abs2$label, mean_abs2$variable)
  
  shap_ren2$variable <- factor(
    shap_ren2$variable,
    levels = mean_abs2$variable
  )
  
  gg_keep2_pretty <- make_shap_beeswarm(
    shap_df = shap_ren2,
    feat_order_labels = feat_order_labels2
  )
  print(gg_keep2_pretty)
}

# -------------------------------------------------
# 13. Boxplots for DIQ010 / SMQ020 categories (optional)
# -------------------------------------------------
label_map_DIQ010 <- function(v) {
  out <- as.character(v)
  out[v == 1] <- "Yes"
  out[v == 2] <- "No"
  out[v == 3] <- "Borderline"
  out
}

if ("DIQ010" %in% shap_long$variable) {
  diq <- shap_long[variable == "DIQ010"]
  diq$rfvalue_cat <- label_map_DIQ010(diq$feat_val)
  
  gg_diq <- ggplot(diq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Doctor told you have diabetes",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_diq)
}

if ("SMQ020" %in% shap_long$variable) {
  smq <- shap_long[variable == "SMQ020"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Yes",
    ifelse(smq$feat_val == 2, "No", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Smoked ≥100 cigarettes in life",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}

# -------------------------------------------------
# 11. Beeswarm for selected features of interest
#     (RIDAGEYR, LBXTC, INDFMPIR, LBXRDW, LBXPLTSI, LBXGH, LBDHDD, BPXOSY1)
# -------------------------------------------------

keep_vars <- c("RIDAGEYR", "LBXTC", "INDFMPIR", "LBXRDW",
               "LBXPLTSI", "LBXGH", "LBDHDD", "BPXOSY1")

# Keep only features that actually exist in shap_long
keep_vars <- keep_vars[keep_vars %in% shap_long$variable]

if (length(keep_vars) > 0) {
  # Subset to selected variables
  shap_keep <- shap_long[variable %in% keep_vars]
  
  # Compute mean absolute SHAP value per variable
  mean_abs <- shap_keep[, .(
    MeanAbsSHAP = mean(abs(shap_val), na.rm = TRUE)
  ), by = variable]
  
  # Preserve the same order as keep_vars
  mean_abs <- mean_abs[match(keep_vars, mean_abs$variable)]
  
  # Create labels with mean SHAP values
  mean_abs[, label := paste0(
    variable, "  ",
    formatC(MeanAbsSHAP, digits = 3, format = "f")
  )]
  
  # Create mapping for labels
  feat_order_labels <- setNames(mean_abs$label, mean_abs$variable)
  
  # Ensure plotting order matches keep_vars
  shap_keep$variable <- factor(shap_keep$variable, levels = keep_vars)
  
  # Generate and print the beeswarm plot
  gg_keep_pretty <- make_shap_beeswarm(
    shap_df = shap_keep,
    feat_order_labels = feat_order_labels
  )
  
  print(gg_keep_pretty)
}


# -------------------------------------------------
# 14. (Optional) Save predictions
# -------------------------------------------------
# out <- data.frame(Actual = truth, Prob_1 = probs, Predicted = pred_cls)
# write.csv(out, "svm_predictions_tuned.csv", row.names = FALSE)
label_map_DIQ010 <- function(v) {
  out <- as.character(v)
  out[v == 1] <- "Less than 9th grade"
  out[v == 2] <- "9-11th grade"
  out[v == 3] <- "High school graduate/GED or equivalent"
  out[v == 4] <- "Some college or AA degree"
  out[v == 5] <- "College graduate or above"
  out
}

if ("DMDEDUC2" %in% shap_long$variable) {
  diq <- shap_long[variable == "DMDEDUC2"]
  diq$rfvalue_cat <- label_map_DIQ010(diq$feat_val)
  
  gg_diq <- ggplot(diq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Education level - Adults 20+",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_diq)
}

if ("BPQ101D" %in% shap_long$variable) {
  smq <- shap_long[variable == "BPQ101D"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Yes",
    ifelse(smq$feat_val == 2, "No", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Taking meds to lower blood cholesterol?",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}
#RIAGENDR
if ("RIAGENDR" %in% shap_long$variable) {
  smq <- shap_long[variable == "RIAGENDR"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Male",
    ifelse(smq$feat_val == 2, "Female", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Gender",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}
#BPQ020
if ("BPQ020" %in% shap_long$variable) {
  smq <- shap_long[variable == "BPQ020"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Yes",
    ifelse(smq$feat_val == 2, "No", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Ever told you had high blood pressure",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}
#BPQ080
if ("BPQ080" %in% shap_long$variable) {
  smq <- shap_long[variable == "BPQ080"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Yes",
    ifelse(smq$feat_val == 2, "No", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Doctor told you - high cholesterol level",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}
#RXQ033
if ("RXQ033" %in% shap_long$variable) {
  smq <- shap_long[variable == "RXQ033"]
  smq$rfvalue_cat <- ifelse(
    smq$feat_val == 1, "Yes",
    ifelse(smq$feat_val == 2, "No", as.character(smq$feat_val))
  )
  
  gg_smq <- ggplot(smq,
                   aes(x = rfvalue_cat,
                       y = shap_val,
                       fill = rfvalue_cat)) +
    geom_boxplot(alpha = 0.7, outlier.colour = "red") +
    labs(x = "Taken prescription medicine, past month",
         y = "SHAP value") +
    theme_minimal() +
    theme(
      legend.position = "none",
      panel.border = element_rect(color = "black", fill = NA, size = 1),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  print(gg_smq)
}


