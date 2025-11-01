# -------------------------------
# Tuned Random Forest (ranger) — 10-fold CV
# Includes confusion matrix figures
# -------------------------------
#install.packages("ranger")
#install.packages(
#  "https://cran.r-project.org/bin/windows/contrib/4.2/ranger_0.16.0.zip",
#  repos = NULL,
#  type = "win.binary"
#)
#suppressPackageStartupMessages({
#  library(ranger)
#  library(caret)
#  library(pROC)
#  library(ggplot2)
#})
library(ranger)
library(caret)
library(pROC)
library(ggplot2)
set.seed(123)

# Safety: close any stray connections that can trigger "invalid connection"
suppressWarnings(try(closeAllConnections(), silent = TRUE))

# ---- Load ----
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\revised_data\\selected_training_merged_file.csv", header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\revised_data\\selected_validation_merged_file.csv", header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSet\\C3E5P0.6\\selected_training_merged_file.csv",
#              header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
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
# ---- Helper: map to valid class names for caret (can't start with number) ----
prepare_target <- function(x) {
  x <- trimws(as.character(x))
  x <- ifelse(x %in% c("Yes","1","TRUE","T"), "Positive",
              ifelse(x %in% c("No","0","FALSE","F"), "Negative", x))
  factor(x, levels = c("Negative","Positive"))
}

df$CVD_risk      <- prepare_target(df$CVD_risk)
df_test$CVD_risk <- prepare_target(df_test$CVD_risk)

# ---- Cast character predictors to factors ----
char_to_factor <- function(d) {
  char_cols <- vapply(d, is.character, logical(1))
  d[char_cols] <- lapply(d[char_cols], factor)
  d
}
df      <- char_to_factor(df)
df_test <- char_to_factor(df_test)

# ---- Common predictors only ----
pred_train <- setdiff(names(df), "CVD_risk")
pred_test  <- setdiff(names(df_test), "CVD_risk")
common_pred <- intersect(pred_train, pred_test)
stopifnot(length(common_pred) > 0)

df_sub      <- df[, c("CVD_risk", common_pred), drop = FALSE]
df_test_sub <- df_test[, c("CVD_risk", common_pred), drop = FALSE]

# ---- Drop near-zero-variance predictors (based on train) ----
nzv_idx <- nearZeroVar(df_sub[, common_pred, drop = FALSE])
if (length(nzv_idx) > 0) {
  drop_vars   <- common_pred[nzv_idx]
  df_sub      <- df_sub[, setdiff(names(df_sub), drop_vars), drop = FALSE]
  df_test_sub <- df_test_sub[, setdiff(names(df_test_sub), drop_vars), drop = FALSE]
  message("Dropped NZV predictors: ", paste(drop_vars, collapse = ", "))
}
final_pred <- setdiff(names(df_sub), "CVD_risk")
p <- length(final_pred)

# ---- Guard numeric columns for Inf/NaN (defensive) ----
num_cols <- names(df_sub)[sapply(df_sub, is.numeric)]
for (cc in num_cols) {
  v <- df_sub[[cc]]
  v[!is.finite(v)] <- median(v[is.finite(v)], na.rm = TRUE)
  df_sub[[cc]] <- v
  v2 <- df_test_sub[[cc]]
  v2[!is.finite(v2)] <- median(v2[is.finite(v2)], na.rm = TRUE)
  df_test_sub[[cc]] <- v2
}

# ---- caret wants the first level to be the positive class for twoClassSummary
# We'll relevel so Positive is first
df_sub$CVD_risk <- stats::relevel(df_sub$CVD_risk, ref = "Positive")

ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,  # gives ROC
  savePredictions = "final",
  verboseIter = FALSE,
  allowParallel = FALSE              # avoid background clusters
)

grid <- expand.grid(
  mtry = unique(pmax(1, round(c(sqrt(p), p/3, p/2)))),
  splitrule = c("gini", "extratrees"),
  min.node.size = c(1, 3, 5)
)

# ---- Try caret::train() first ----
run_caret <- function() {
  train(
    CVD_risk ~ .,
    data = df_sub,
    method = "ranger",
    metric = "ROC",          # maximize AUC
    trControl = ctrl,
    tuneGrid = grid,
    num.trees = 1000,
    importance = "impurity"
  )
}
rf_cv <- try(run_caret(), silent = TRUE)

if (inherits(rf_cv, "try-error")) {
  message("caret::train() failed (connection issue). Falling back to manual 10-fold CV tuner...")
  
  # Manual stratified folds
  folds <- createFolds(df_sub$CVD_risk, k = 10, returnTrain = TRUE)
  combos <- grid
  combos$ROC <- NA_real_
  
  for (i in seq_len(nrow(combos))) {
    mtry_i <- combos$mtry[i]
    split_i  <- as.character(combos$splitrule[i])
    mns_i    <- combos$min.node.size[i]
    aucs <- c()
    
    for (f in seq_along(folds)) {
      tr_idx <- folds[[f]]
      te_idx <- setdiff(seq_len(nrow(df_sub)), tr_idx)
      tr <- df_sub[tr_idx, , drop = FALSE]
      te <- df_sub[te_idx, , drop = FALSE]
      
      fit <- ranger(
        CVD_risk ~ .,
        data = tr,
        probability = TRUE,
        num.trees = 1000,
        mtry = mtry_i,
        splitrule = split_i,
        min.node.size = mns_i,
        importance = "impurity",
        seed = 123
      )
      
      # Probability for Positive
      pr <- predict(fit, data = te, type = "response")$predictions[, "Positive"]
      # AUC fold
      fold_auc <- as.numeric(pROC::auc(pROC::roc(te$CVD_risk, pr,
                                                 levels = c("Negative","Positive"))))
      aucs <- c(aucs, fold_auc)
    }
    
    combos$ROC[i] <- mean(aucs, na.rm = TRUE)
  }
  
  best_row  <- which.max(combos$ROC)
  bestTune  <- combos[best_row, c("mtry","splitrule","min.node.size")]
  print(bestTune)
  cat(sprintf("Best mean CV AUC: %.4f\n", combos$ROC[best_row]))
  
  # Final train on all training data with best params
  rf_fit <- ranger(
    CVD_risk ~ .,
    data = df_sub,
    probability = TRUE,
    num.trees = 1000,
    mtry = bestTune$mtry,
    splitrule = as.character(bestTune$splitrule),
    min.node.size = bestTune$min.node.size,
    importance = "impurity",
    seed = 123
  )
  
  rf_obj <- list(finalModel = rf_fit, bestTune = bestTune)
  class(rf_obj) <- "manual_rf"
  
} else {
  cat("\n=== RF Tuning (10-fold CV via caret) — Best hyperparameters ===\n")
  print(rf_cv$bestTune)
  rf_obj <- rf_cv
}

# ---- Predict on held-out test ----
if (inherits(rf_obj, "manual_rf")) {
  rf_fit  <- rf_obj$finalModel
  rf_pred <- predict(rf_fit, data = df_test_sub, type = "response")
  probs   <- as.numeric(rf_pred$predictions[, "Positive"])
} else {
  rf_pred <- predict(rf_obj, newdata = df_test_sub, type = "prob")
  probs   <- as.numeric(rf_pred[["Positive"]])
}

pred_cls <- factor(
  ifelse(probs >= 0.5, "Positive", "Negative"),
  levels = c("Negative","Positive")
)
truth <- df_test_sub$CVD_risk

# ---- Metrics on test ----
cm <- confusionMatrix(pred_cls, truth, positive = "Positive", mode = "everything")
print(cm)

roc_obj <- roc(response = truth, predictor = probs,
               levels = c("Negative","Positive"))
auc_val <- auc(roc_obj)
cat(sprintf("AUC: %.4f\n", as.numeric(auc_val)))

tp <- as.numeric(cm$table["Positive","Positive"])
tn <- as.numeric(cm$table["Negative","Negative"])
fp <- as.numeric(cm$table["Positive","Negative"])
fn <- as.numeric(cm$table["Negative","Positive"])
mcc <- (tp*tn - fp*fn) /
  sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))
cat(sprintf("MCC: %.4f\n", mcc))

accuracy_val <- sum(pred_cls == truth) / length(truth)
cat(sprintf("Accuracy: %.4f\n", accuracy_val))

# ---- ROC plot ----
plot(roc_obj,
     main = sprintf("ROC — Tuned Random Forest (AUC = %.3f)", auc_val))

# ---- Variable importance plot (top 20) ----
if (inherits(rf_obj, "manual_rf")) {
  vi <- rf_obj$finalModel$variable.importance
  imp_df <- data.frame(Variable = names(vi),
                       Importance = as.numeric(vi))
} else {
  imp <- varImp(rf_obj)$importance
  imp$Variable <- rownames(imp)
  names(imp)[1] <- "Importance"
  imp_df <- imp[, c("Variable","Importance")]
}
imp_df <- imp_df[order(-imp_df$Importance), ]
topn <- min(20, nrow(imp_df))

ggplot(imp_df[1:topn, ],
       aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Random Forest Variable Importance (Top 20)",
       x = "Feature", y = "Importance") +
  theme_minimal()

# ============================
# Confusion matrix plots (match XGBoost style)
# ============================

draw_confusion_matrix <- function(cm,
                                  neg_label = "Negative",
                                  pos_label = "Positive") {
  # cm$table is Prediction x Reference (Predicted rows, Actual cols)
  # We'll grab counts in the same order your xgboost code assumed: res[1:4]
  # res[1] -> TN, res[2] -> FP, res[3] -> FN, res[4] -> TP
  # We'll compute them explicitly to be safe:
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table[pos_label, neg_label])
  FN <- as.numeric(cm$table[neg_label, pos_label])
  TP <- as.numeric(cm$table[pos_label, pos_label])
  res <- c(TN, FP, FN, TP)
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  # same coordinate system as your xgb version (no plot(), just draw rect / text)
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  
  text(250, 405, "Reference", cex=1.3, font=2)
  text(175, 375, "Prediction", cex=1.3, srt=90, font=2)
  
  # row/col labels
  text(250, 375, neg_label, cex=1.3, font=2)
  text(250, 350, pos_label, cex=1.3, font=2)
  text(300, 400, neg_label, cex=1.3, font=2)
  text(400, 400, pos_label, cex=1.3, font=2)
  
  # cell values, same positions you used
  text(300, 375, res[1], cex=1.6, font=2)  # TN
  text(400, 375, res[2], cex=1.6, font=2)  # FP
  text(300, 350, res[3], cex=1.6, font=2)  # FN
  text(400, 350, res[4], cex=1.6, font=2)  # TP
}

draw_confusion_matrix2 <- function(cm,
                                   neg_label = "Negative",
                                   pos_label = "Positive") {
  # same counts as above
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table[pos_label, neg_label])
  FN <- as.numeric(cm$table[neg_label, pos_label])
  TP <- as.numeric(cm$table[pos_label, pos_label])
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  # reproduce your second style exactly:
  plot(c(123, 345), c(300, 452),
       type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  
  rect(150, 430, 240, 370, col='#AF97D0'); text(195, 435, neg_label, cex=1.2)
  rect(250, 430, 340, 370, col='#A7AD50'); text(295, 435, pos_label, cex=1.2)
  
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  
  rect(150, 305, 240, 365, col='#A7AD50')
  rect(250, 305, 340, 365, col='#AF97D0')
  
  text(140, 400, neg_label, cex=1.2, srt=90)
  text(140, 335, pos_label, cex=1.2, srt=90)
  
  # fill in the four cells in SAME order/locations you used before
  text(195, 400, TN, cex=1.6, font=2)  # TN
  text(195, 335, FP, cex=1.6, font=2)  # FP
  text(295, 400, FN, cex=1.6, font=2)  # FN
  text(295, 335, TP, cex=1.6, font=2)  # TP
}

# After you compute cm <- confusionMatrix(...):
draw_confusion_matrix(cm)
draw_confusion_matrix2(cm)
