# -------------------------------
# Tuned SVM (RBF) — 10-fold CV
# Includes confusion matrix figures
# -------------------------------

suppressPackageStartupMessages({
  library(e1071)   # svm, tune.svm
  library(caret)   # confusionMatrix, nearZeroVar, createFolds
  library(pROC)    # roc, auc
})

set.seed(123)

# ---- File paths ----
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\revised_data\\selected_training_merged_file.csv",header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\revised_data\\selected_validation_merged_file.csv",header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\revised_data\\C3E5P0.6\\selected_training_merged_file.csv",
#            header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\revised_data\\C3E5P0.6\\selected_validation_merged_file.csv",
#                   header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)


df <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin5\\data50\\selected_training_merged_file.csv",
               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
df_test <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\revised_data\\bin5\\data50\\selected_validation_merged_file.csv",
                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

#df <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\pearsonAndChi-squared\\selected_training_merged_file.csv",
#               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\pearsonAndChi-squared\\selected_validation_merged_file.csv",
#                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

# ---- Helper: prepare target as factor {0,1} ----
prepare_target <- function(x) {
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

stopifnot("CVD_risk" %in% names(df), "CVD_risk" %in% names(df_test))

# ---- Targets ----
df$CVD_risk      <- prepare_target(df$CVD_risk)
df_test$CVD_risk <- prepare_target(df_test$CVD_risk)

# ---- Convert character predictors to factors ----
char_to_factor <- function(d) {
  char_cols <- vapply(d, is.character, logical(1))
  d[char_cols] <- lapply(d[char_cols], factor)
  d
}
df      <- char_to_factor(df)
df_test <- char_to_factor(df_test)

# ---- Keep only predictors present in BOTH ----
pred_train <- setdiff(names(df), "CVD_risk")
pred_test  <- setdiff(names(df_test), "CVD_risk")
common_pred <- intersect(pred_train, pred_test)
if (length(common_pred) == 0) stop("No common predictors between train and test.")

df_sub      <- df[, c("CVD_risk", common_pred), drop = FALSE]
df_test_sub <- df_test[, c("CVD_risk", common_pred), drop = FALSE]

# ---- Drop near-zero-variance predictors (based on train) ----
nzv_idx <- nearZeroVar(df_sub[, common_pred, drop = FALSE], saveMetrics = FALSE)
if (length(nzv_idx) > 0) {
  drop_vars   <- common_pred[nzv_idx]
  df_sub      <- df_sub[, setdiff(names(df_sub), drop_vars), drop = FALSE]
  df_test_sub <- df_test_sub[, setdiff(names(df_test_sub), drop_vars), drop = FALSE]
  message("Dropped NZV predictors: ", paste(drop_vars, collapse = ", "))
}
final_pred <- setdiff(names(df_sub), "CVD_risk")

# ---- Defensive cleanup for numeric columns (Inf/NaN)
num_cols <- names(df_sub)[sapply(df_sub, is.numeric)]
for (cc in num_cols) {
  v <- df_sub[[cc]]
  v[!is.finite(v)] <- median(v[is.finite(v)], na.rm = TRUE)
  df_sub[[cc]] <- v
  
  v2 <- df_test_sub[[cc]]
  v2[!is.finite(v2)] <- median(v2[is.finite(v2)], na.rm = TRUE)
  df_test_sub[[cc]] <- v2
}

# ============================
# 10-fold hyperparameter tuning
# ============================

# We'll tune cost (C) and gamma for the RBF kernel.
# Note: keep the grid moderate so runtime is not insane.
cost_grid  <- 2 ^ (-1:3)     # 0.5,1,2,4,8
gamma_grid <- 2 ^ (-5:0)     # 1/32 ... 1

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
  tunecontrol = tune.control(cross = 10)  # <-- 10-fold CV
)

cat("\n=== SVM tuning results (10-fold CV) ===\n")
print(svm_tuned$best.parameters)

# svm_tuned$best.model is already retrained on the full training set
svm_fit <- svm_tuned$best.model

# ============================
# Evaluate on external test set
# ============================

svm_pred <- predict(svm_fit, newdata = df_test_sub, probability = TRUE)
svm_probs_mat <- attr(svm_pred, "probabilities")

if (is.null(svm_probs_mat)) {
  stop("SVM probabilities not available even after tuning.")
}

# Get prob for positive ("1")
pos_col <- which(colnames(svm_probs_mat) == "1")
if (length(pos_col) != 1) {
  truth_levels <- levels(df_test_sub$CVD_risk)
  pos_col <- which(colnames(svm_probs_mat) == truth_levels[length(truth_levels)])
  if (length(pos_col) != 1) stop("Could not identify positive-class probability column after tuning.")
}
probs <- as.numeric(svm_probs_mat[, pos_col])

# Pred class from threshold 0.5
pred_cls <- factor(ifelse(probs >= 0.5, "1", "0"), levels = c("0","1"))
truth    <- df_test_sub$CVD_risk

# Confusion matrix
cm <- confusionMatrix(pred_cls, truth, positive = "1", mode = "everything")
print(cm)

# ROC / AUC
roc_obj  <- roc(response = truth, predictor = probs, levels = c("0","1"), direction = "<")
auc_val  <- auc(roc_obj)
cat(sprintf("SVM AUC (test): %.4f\n", as.numeric(auc_val)))

# MCC
tp <- as.numeric(cm$table["1","1"])
tn <- as.numeric(cm$table["0","0"])
fp <- as.numeric(cm$table["1","0"])
fn <- as.numeric(cm$table["0","1"])
mcc <- (tp*tn - fp*fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
cat(sprintf("MCC: %.4f\n", mcc))

# Accuracy
accuracy_val <- sum(pred_cls == truth) / length(truth)
cat(sprintf("Accuracy: %.4f\n", accuracy_val))

# ROC plot
plot(roc_obj, main = sprintf("ROC — Tuned SVM (RBF) (AUC = %.3f)", auc_val))

# ============================
# Confusion matrix figures (same style as XGBoost)
# ============================

# Style 1: "Reference / Prediction"
draw_confusion_matrix <- function(cm,
                                  neg_label = "0",
                                  pos_label = "1") {
  # Get the four cells explicitly so order is exactly like your xgboost plot
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table[pos_label, neg_label])
  FN <- as.numeric(cm$table[neg_label, pos_label])
  TP <- as.numeric(cm$table[pos_label, pos_label])
  res <- c(TN, FP, FN, TP)
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  # same geometry you used before
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  
  text(250, 405, "Reference",  cex=1.3, font=2)
  text(175, 375, "Prediction", cex=1.3, srt=90, font=2)
  
  # axis class labels
  text(250, 375, neg_label, cex=1.3, font=2)
  text(250, 350, pos_label, cex=1.3, font=2)
  text(300, 400, neg_label, cex=1.3, font=2)
  text(400, 400, pos_label, cex=1.3, font=2)
  
  # counts
  text(300, 375, res[1], cex=1.6, font=2)  # TN
  text(400, 375, res[2], cex=1.6, font=2)  # FP
  text(300, 350, res[3], cex=1.6, font=2)  # FN
  text(400, 350, res[4], cex=1.6, font=2)  # TP
}

# Style 2: "Predicted / Actual" with the 4-quadrant plot
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

# Draw both confusion matrix figures for SVM
draw_confusion_matrix(cm, neg_label = "0", pos_label = "1")
draw_confusion_matrix2(cm, neg_label = "0", pos_label = "1")

# ============================
# (Optional) Save predictions
# ============================
# out <- data.frame(Actual = truth, Prob_1 = probs, Predicted = pred_cls)
# write.csv(out, "svm_predictions_tuned.csv", row.names = FALSE)
