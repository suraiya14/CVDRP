# -------------------------------
# Tuned XGBoost (binary:logistic) — 10-fold CV + Early Stopping
# Confusion matrix: "Negative"/"Positive" (correct FP/FN)
# ROC/AUC: 0/1 orientation
# Includes base-R CM plots + SHAP
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

# ============================================
# 1) Load your data (uncomment ONE pair)
# ============================================
df <- read.csv("D:\\Research_Work\\research on new idae\\idea1(CVD)\\Feature_Extraction\\ADT\\revised_data\\selected_training_merged_file.csv",
               header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)
df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1(CVD)\\Feature_Extraction\\ADT\\revised_data\\selected_validation_merged_file.csv",
                    header = TRUE, check.names = TRUE, stringsAsFactors = FALSE)

stopifnot(exists("df"), exists("df_test"),
          "CVD_risk" %in% names(df), "CVD_risk" %in% names(df_test))

# ============================================
# 2) Target to {0,1} numeric (robust)
# ============================================
to01 <- function(x){
  if (is.factor(x) || is.character(x)) {
    x <- trimws(as.character(x))
    as.numeric(x %in% c("Yes","1","TRUE","T"))
  } else if (is.logical(x)) {
    as.numeric(x)
  } else {
    as.numeric(as.numeric(x) >= 1)
  }
}
df$CVD_risk      <- to01(df$CVD_risk)
df_test$CVD_risk <- to01(df_test$CVD_risk)

# ============================================
# 3) Common features
# ============================================
features <- intersect(setdiff(names(df), "CVD_risk"),
                      setdiff(names(df_test), "CVD_risk"))
stopifnot(length(features) > 0)

X_train <- data.matrix(df[, features, drop=FALSE])
y_train <- as.numeric(df$CVD_risk)
X_test  <- data.matrix(df_test[, features, drop=FALSE])
y_test  <- as.numeric(df_test$CVD_risk)

dtrain <- xgb.DMatrix(X_train, label=y_train)
dtest  <- xgb.DMatrix(X_test,  label=y_test)

# ============================================
# 4) Class imbalance
# ============================================
pos <- sum(y_train == 1); neg <- sum(y_train == 0)
scale_pos_weight <- if (pos > 0) neg/pos else 1

# ============================================
# 5) 10-fold CV + early stopping on AUC
# ============================================
param_grid <- expand.grid(
  eta              = c(0.03, 0.06),
  max_depth        = c(3, 5, 7),
  subsample        = c(0.8, 1.0),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 3),
  reg_lambda       = c(1, 2),
  KEEP.OUT.ATTRS   = FALSE, stringsAsFactors = FALSE
)

best_auc <- -Inf; best_iter <- NA_integer_; best_param <- NULL
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
    params=params, data=dtrain, nrounds=5000, nfold=10,
    stratified=TRUE, early_stopping_rounds=100,
    maximize=TRUE, verbose=0
  )
  mean_auc <- cv$evaluation_log$test_auc_mean[cv$best_iteration]
  if (mean_auc > best_auc) {
    best_auc  <- mean_auc
    best_iter <- cv$best_iteration
    best_param <- params
  }
}
cat(sprintf("Best CV AUC: %.5f at %d rounds\n", best_auc, best_iter))
cat("Best params:\n"); print(best_param)

# ============================================
# 6) Final train
# ============================================
fit_xgb <- xgb.train(params=best_param, data=dtrain,
                     nrounds=best_iter,
                     watchlist=list(train=dtrain, test=dtest),
                     verbose=0)

# ============================================
# 7) Predictions & metrics
# ============================================
probs <- predict(fit_xgb, dtest)

# Threshold -> human labels
pred_lbl  <- factor(ifelse(probs >= 0.5, "Positive", "Negative"),
                    levels=c("Negative","Positive"))
truth_lbl <- factor(ifelse(y_test == 1, "Positive", "Negative"),
                    levels=c("Negative","Positive"))

# Confusion matrix (caret: rows=Prediction, cols=Reference)
xgb_cm <- caret::confusionMatrix(pred_lbl, truth_lbl,
                                 positive="Positive", mode="everything")
print(xgb_cm)
print(xgb_cm$byClass)

# ROC/AUC (keep 0/1 orientation)
roc_curve <- pROC::roc(response=factor(y_test, levels=c(0,1)),
                       predictor=probs, levels=c("0","1"), direction="<")
auc_value <- pROC::auc(roc_curve)
cat(sprintf("XGBoost AUC: %.4f\n", as.numeric(auc_value)))
plot(roc_curve, main=sprintf("ROC — Tuned XGBoost (AUC = %.3f)", auc_value))

# MCC with correct FP/FN indices (rows=Prediction, cols=Reference)
TT <- xgb_cm$table
tp <- as.numeric(TT["Positive","Positive"])
tn <- as.numeric(TT["Negative","Negative"])
fp <- as.numeric(TT["Positive","Negative"])  # predicted Positive, actual Negative
fn <- as.numeric(TT["Negative","Positive"])  # predicted Negative, actual Positive
mcc <- (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
cat(sprintf("MCC: %.4f\n", mcc))

# ============================================
# 8) Base R confusion-matrix plots (corrected)
# ============================================
draw_confusion_matrix <- function(cm,
                                  neg_label="Negative",
                                  pos_label="Positive") {
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table["Positive", neg_label])  # correct
  FN <- as.numeric(cm$table[neg_label, "Positive"])  # correct
  TP <- as.numeric(cm$table["Positive","Positive"])
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  
  text(250, 405, "Actual",     cex=1.3, font=2)
  text(175, 375, "Predicted",  cex=1.3, srt=90, font=2)
  
  text(250, 375, neg_label, cex=1.3, font=2)
  text(250, 350, pos_label, cex=1.3, font=2)
  text(300, 400, neg_label, cex=1.3, font=2)
  text(400, 400, pos_label, cex=1.3, font=2)
  
  text(300, 375, TN, cex=1.6, font=2)  # TN
  text(400, 375, FP, cex=1.6, font=2)  # FP
  text(300, 350, FN, cex=1.6, font=2)  # FN
  text(400, 350, TP, cex=1.6, font=2)  # TP
}

draw_confusion_matrix2 <- function(cm,
                                   neg_label="Negative",
                                   pos_label="Positive") {
  TN <- as.numeric(cm$table[neg_label, neg_label])
  FP <- as.numeric(cm$table["Positive", neg_label])  # correct
  FN <- as.numeric(cm$table[neg_label, "Positive"])  # correct
  TP <- as.numeric(cm$table["Positive","Positive"])
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  plot(c(123, 345), c(300, 452), type="n", xlab="", ylab="", xaxt='n', yaxt='n')
  
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

# Draw
draw_confusion_matrix(xgb_cm)
draw_confusion_matrix2(xgb_cm)

# ============================================
# 9) SHAP (same outputs you had)
# ============================================
X_sample <- X_train  # optionally subsample for speed

shap_values <- SHAPforxgboost::shap.prep(xgb_model=fit_xgb, X_train=X_sample)

# Beeswarm (all features)
shap_beeswarm_plot <- SHAPforxgboost::shap.plot.summary(shap_values) +
  theme(axis.title.y = element_text(face="bold"))
print(shap_beeswarm_plot)

# Mean |SHAP| bar
shap_bar_data <- SHAPforxgboost::shap.importance(shap_values)
colnames(shap_bar_data) <- c("Variable", "Mean_SHAP")
shap_bar_plot <- ggplot(shap_bar_data,
                        aes(x=reorder(Variable, Mean_SHAP), y=Mean_SHAP)) +
  geom_bar(stat="identity", fill="steelblue") +
  coord_flip() +
  labs(x="Feature", y="Mean |SHAP value|") +
  theme_minimal() +
  theme(axis.title = element_text(face="bold"),
        panel.border = element_rect(color="black", fill=NA, size=1),
        panel.background = element_blank())
print(shap_bar_plot)

# Optional: rename and replot a subset
shap_dt <- as.data.table(shap_values)
shap_dt[variable == "RIDAGEYR", variable := "Suraiya"]
shap_dt[variable == "LBXTC",    variable := "Kayes"]
sub <- shap_dt[variable %in% c("Suraiya","Kayes")]
if (nrow(sub) > 0) {
  sub[, variable := factor(variable, levels=c("Suraiya","Kayes"))]
  print(SHAPforxgboost::shap.plot.summary(sub) +
          theme(axis.title.y = element_text(face="bold")))
}

# ============================
# (Optional) Save probabilities
# ============================
# write.csv(data.frame(Actual=y_test, Prob_1=probs,
#                      Predicted=ifelse(probs>=0.5,1,0)),
#           "probabilities_xgb.csv", row.names=FALSE)
