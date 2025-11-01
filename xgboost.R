# -------------------------------
# Tuned XGBoost (binary:logistic) — 10-fold CV + Early Stopping
# Keeps all original outputs & plots
# -------------------------------

suppressPackageStartupMessages({
  library(caret)
  library(pROC)
  library(xgboost)
  library(SHAPforxgboost)
  library(data.table)
  library(ggplot2)
  library(stringr)
})

set.seed(123)

# ---- Read training and test datasets ----
# ---- Load ----
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSet\\ADT\\selected_training_merged_file.csv",
               #header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSet\\ADT\\selected_validation_merged_file.csv",
                   # header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSet\\C3E5P0.6\\selected_training_merged_file.csv",
#               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSet\\C3E5P0.6\\selected_validation_merged_file.csv",
#                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

#df <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin10\\data75\\selected_training_merged_file.csv",
#               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin10\\data75\\selected_validation_merged_file.csv",
#                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

df <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\pearsonAndChi-squared\\selected_training_merged_file.csv",
               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
df_test <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\pearsonAndChi-squared\\selected_validation_merged_file.csv",
                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

# ---- Target to {0,1} numeric ----
df$CVD_risk     <- ifelse(df$CVD_risk == 'Yes', 1, 0)
df_test$CVD_risk<- ifelse(df_test$CVD_risk == 'Yes', 1, 0)

# ---- Use only features present in BOTH train & test ----
features <- intersect(setdiff(names(df), "CVD_risk"), setdiff(names(df_test), "CVD_risk"))

# ---- Matrices (keep your data.matrix approach to preserve original names for SHAP) ----
X_train <- data.matrix(df[, features, drop = FALSE])
y_train <- as.numeric(df$CVD_risk)
X_test  <- data.matrix(df_test[, features, drop = FALSE])
y_test  <- as.numeric(df_test$CVD_risk)

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

# ---- Optional imbalance handling ----
pos <- sum(y_train == 1); neg <- sum(y_train == 0)
scale_pos_weight <- if (pos > 0) neg/pos else 1

# ---- Small but strong tuning grid (10-fold CV + early stopping on AUC) ----
param_grid <- expand.grid(
  eta              = c(0.03, 0.06),
  max_depth        = c(3, 5, 7),
  subsample        = c(0.8, 1.0),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 3),
  reg_lambda       = c(1, 2),
  KEEP.OUT.ATTRS   = FALSE, stringsAsFactors = FALSE
)

best_auc   <- -Inf
best_iter  <- NA_integer_
best_param <- NULL

cat("Starting 10-fold CV over", nrow(param_grid), "settings...\n")
for (i in seq_len(nrow(param_grid))) {
  p <- as.list(param_grid[i, ])
  
  params <- list(
    objective         = "binary:logistic",
    eval_metric       = "auc",
    eta               = p$eta,
    max_depth         = p$max_depth,
    subsample         = p$subsample,
    colsample_bytree  = p$colsample_bytree,
    min_child_weight  = p$min_child_weight,
    reg_lambda        = p$reg_lambda,
    scale_pos_weight  = scale_pos_weight,
    tree_method       = "hist"
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 5000,
    nfold = 10,                # <-- 10-fold per your request
    stratified = TRUE,
    early_stopping_rounds = 100,
    maximize = TRUE,
    verbose = 0
  )
  
  mean_auc <- cv$evaluation_log$test_auc_mean[cv$best_iteration]
  if (mean_auc > best_auc) {
    best_auc   <- mean_auc
    best_iter  <- cv$best_iteration
    best_param <- params
  }
}

cat(sprintf("Best CV AUC: %.5f at %d rounds\n", best_auc, best_iter))
cat("Best params:\n"); print(best_param)

# ---- Final training with best params/rounds ----
fit_xgb <- xgb.train(
  params  = best_param,
  data    = dtrain,
  nrounds = best_iter,
  watchlist = list(train = dtrain, test = dtest),
  verbose = 0
)

# =======================
# Predictions & metrics
# =======================
probs <- predict(fit_xgb, dtest)

# (Your original code mixed types; keep factor here for CM)
pred_labels <- ifelse(probs > 0.5, 1, 0)
pred_fac    <- factor(pred_labels, levels = c(0,1))
truth_fac   <- factor(y_test,     levels = c(0,1))

# Confusion matrix (everything)
xgbconfusion <- caret::confusionMatrix(pred_fac, truth_fac, positive = '1', mode = "everything")
print(xgbconfusion)
print(xgbconfusion$byClass)

# AUC + ROC plot
roc_curve <- pROC::roc(truth_fac, probs, levels = c("0","1"), direction = "<")
auc_value <- pROC::auc(roc_curve)
cat(sprintf("XGBoost AUC: %.4f\n", as.numeric(auc_value)))
plot(roc_curve, main = sprintf("ROC — Tuned XGBoost (AUC = %.3f)", auc_value))

# MCC (same as your logic)
TT <- table(Actual = truth_fac, Predicted = pred_fac)
tp <- as.numeric(TT["1","1"])
tn <- as.numeric(TT["0","0"])
fp <- as.numeric(TT["0","1"])
fn <- as.numeric(TT["1","0"])
mcc <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
cat(sprintf("MCC: %.4f\n", mcc))

# =================================
# SHAP — preserve all your outputs
# =================================
X_sample <- X_train  # you can subsample rows if large
shap_values <- SHAPforxgboost::shap.prep(xgb_model = fit_xgb, X_train = X_sample)

# 1) SHAP beeswarm (all features)
shap_beeswarm_plot <- SHAPforxgboost::shap.plot.summary(shap_values) +
  theme(axis.title.y = element_text(face = "bold"))
print(shap_beeswarm_plot)

# 2) SHAP bar (mean |SHAP|)
shap_bar_data <- SHAPforxgboost::shap.importance(shap_values)
colnames(shap_bar_data) <- c("Variable", "Mean_SHAP") # ensure consistent names
shap_bar_plot <- ggplot(shap_bar_data, aes(x = reorder(Variable, Mean_SHAP), y = Mean_SHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "Feature", y = "Mean |SHAP value|") +
  theme_minimal() +
  theme(axis.title = element_text(face = "bold"),
        panel.border = element_rect(color = "black", fill = NA, size = 1),
        panel.background = element_blank())
print(shap_bar_plot)

# Helper: standardize SHAP column names for 'shap value' and 'feature value'
std_shap_cols <- function(dt) {
  # Create 'shap_val' and 'feat_val' regardless of package column naming
  if ("shap_value" %in% names(dt))   dt[, shap_val := shap_value]
  if ("phi" %in% names(dt))          dt[, shap_val := phi]
  if ("value" %in% names(dt))        dt[, shap_val := value]  # fallback if value is shap
  if (!("shap_val" %in% names(dt)))  stop("Cannot find SHAP value column in shap_values.")
  
  if ("X_value" %in% names(dt))      dt[, feat_val := X_value]
  if ("rfvalue" %in% names(dt))      dt[, feat_val := rfvalue]
  if ("value" %in% names(dt) && !("feat_val" %in% names(dt))) dt[, feat_val := value]
  if (!("feat_val" %in% names(dt)))  dt[, feat_val := NA_real_]
  
  dt
}

shap_dt <- data.table::as.data.table(shap_values)
shap_dt <- std_shap_cols(shap_dt)

# 3) Beeswarm for selected features
keep_vars <- c("RIDAGEYR", "LBXTC", "LBXGH", "BPXOSY1")
shap_dt_sub <- shap_dt[variable %in% keep_vars]
shap_dt_sub[, variable := factor(variable, levels = keep_vars)]
SHAPforxgboost::shap.plot.summary(shap_dt_sub) +
  theme(axis.title.y = element_text(face = "bold"))

# 4) Rename variables and plot again
shap_dt2 <- data.table::copy(shap_dt)
shap_dt2[variable == "RIDAGEYR", variable := "Suraiya"]
shap_dt2[variable == "LBXTC",    variable := "Kayes"]
shap_dt2_sub <- shap_dt2[variable %in% c("Suraiya", "Kayes")]
shap_dt2_sub[, variable := factor(variable, levels = c("Suraiya", "Kayes"))]
SHAPforxgboost::shap.plot.summary(shap_dt2_sub) +
  theme(axis.title.y = element_text(face = "bold"))

# 5) Boxplots: DIQ010 (Yes/No/Borderline) and SMQ020 (Yes/No)
#    We use 'feat_val' (standardized) for categories and 'shap_val' for SHAP y-axis.
#    If your dataset uses codes (1/2/3), we map them to labels.

# DIQ010
smq_shap <- shap_dt[variable == "DIQ010"]
smq_shap[, rfvalue_cat := as.character(feat_val)]
smq_shap[feat_val == 1, rfvalue_cat := "Yes"]
smq_shap[feat_val == 2, rfvalue_cat := "No"]
smq_shap[feat_val == 3, rfvalue_cat := "Borderline"]

gg_smq <- ggplot(smq_shap, aes(x = rfvalue_cat, y = shap_val, fill = rfvalue_cat)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  labs(x = "Doctor told you have diabetes (DIQ010)", y = "SHAP value") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, size = 1),
        panel.background = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1))
print(gg_smq)

# SMQ020
smq_shap2 <- shap_dt[variable == "SMQ020"]
smq_shap2[, rfvalue_cat := as.character(feat_val)]
smq_shap2[feat_val == 1, rfvalue_cat := "Yes"]
smq_shap2[feat_val == 2, rfvalue_cat := "No"]

gg_smq2 <- ggplot(smq_shap2, aes(x = rfvalue_cat, y = shap_val, fill = rfvalue_cat)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  labs(x = "Smoked ≥100 cigarettes in life (SMQ020)", y = "SHAP value") +
  theme_minimal() +
  theme(legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, size = 1),
        panel.background = element_blank(),
        axis.text.x = element_text(angle = 30, hjust = 1))
print(gg_smq2)

# ============================
# Your base R confusion plots
# ============================
draw_confusion_matrix <- function(cm) {
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  text(250, 405, "Reference", cex=1.3, font=2)
  text(175, 375, "Prediction", cex=1.3, srt=90, font=2)
  res <- as.numeric(cm$table)
  text(250, 375, "-1", cex=1.3, font=2, col='black')
  text(250, 350, "1",  cex=1.3, font=2, col='black')
  text(300, 400, "-1", cex=1.3, font=2, col='black')
  text(300, 375, res[1], cex=1.6, font=2, col='black')
  text(300, 350, res[3], cex=1.6, font=2, col='black')
  text(400, 400, "1",  cex=1.3, font=2, col='black')
  text(400, 375, res[2], cex=1.6, font=2, col='black')
  text(400, 350, res[4], cex=1.6, font=2, col='black')
}

draw_confusion_matrix2 <- function(cm) {
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  plot(c(123, 345), c(300, 452), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  rect(150, 430, 240, 370, col='#AF97D0'); text(195, 435, -1, cex=1.2)
  rect(250, 430, 340, 370, col='#A7AD50'); text(295, 435,  1, cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual',    cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#A7AD50')
  rect(250, 305, 340, 365, col='#AF97D0')
  text(140, 400, -1, cex=1.2, srt=90)
  text(140, 335,  1, cex=1.2, srt=90)
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='black')
  text(195, 335, res[2], cex=1.6, font=2, col='black')
  text(295, 400, res[3], cex=1.6, font=2, col='black')
  text(295, 335, res[4], cex=1.6, font=2, col='black')
}

# Draw both, as in your script:
draw_confusion_matrix(xgbconfusion)
draw_confusion_matrix2(xgbconfusion)

# ============================
# (Optional) Save probabilities
# ============================
# write.csv(probs, "D:\\Research_Work\\Disertation Project 3\\RawData\\FeatureExtraction\\CVFS\\probability\\probability_C2E10P0.4.csv", row.names = FALSE)
