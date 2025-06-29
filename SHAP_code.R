# Load libraries
library(doParallel)
library(e1071)
library(caret)
library(ROCR)
library(mltools)
library(pROC)
library(parallel)
library(ggplot2)
library(tidyr)
library(xgboost)
library(SHAPforxgboost)
library(viridis) # For scale_color_viridis_c

# Read training and test datasets
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\selected_training_merged_file.csv", header = TRUE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\selected_validation_merged_file.csv", header = TRUE)
df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\C2E10P0.6/selected_training_merged_file.csv", header = TRUE)
df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\C2E10P0.6\\selected_validation_merged_file.csv", header = TRUE)
#df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\hypergraph\\bin10\\data25/selected_training_merged_file.csv", header = TRUE)
#df_test <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\hypergraph\\bin10\\data25\\selected_validation_merged_file.csv", header = TRUE)

set.seed(123)
# Convert target variable to binary numeric
df$CVD_risk <- ifelse(df$CVD_risk == 'Yes', 1, 0)
df_test$CVD_risk <- ifelse(df_test$CVD_risk == 'Yes', 1, 0)

# Define feature columns
features <- setdiff(colnames(df), "CVD_risk")

# Prepare train and test matrices
X_train <- data.matrix(df[, features])
y_train <- as.numeric(df$CVD_risk)
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

X_test <- data.matrix(df_test[, features])
y_test <- as.numeric(df_test$CVD_risk)
dvalid <- xgb.DMatrix(data = X_test, label = y_test)

# Set XGBoost parameters
params <- list(
  objective = "binary:logistic",
  learning_rate = 0.05,
  subsample = 0.9,
  colsample_bynode = 1,
  reg_lambda = 2,
  max_depth = 5
)

# Train XGBoost with parallel processing
cores <- detectCores()
cl <- makePSOCKcluster(cores - 2)
registerDoParallel(cl)

fit_xgb <- xgb.train(
  params = params,
  data = dtrain,
  watchlist = list(validation = dvalid),
  nrounds = 10000,
  print_every_n = 100
)

stopCluster(cl)  # Stop the parallel cluster after training

# Compute SHAP values on entire training set
X_sample <- X_train  # Or subset if needed: X_train[sample(nrow(X_train), 5000), ]
shap_values <- shap.prep(xgb_model = fit_xgb, X_train = X_sample)

# Check column names to confirm what you have in shap_values:
print(head(shap_values))
print(colnames(shap_values))
# Typical column names: variable, shap_value, X_value

# SHAP beeswarm plot (all features)
shap_beeswarm_plot <- shap.plot.summary(shap_values) +
  theme(axis.title.y = element_text(face = "bold"))
print(shap_beeswarm_plot)



shap_bar_data <- shap.importance(shap_values)
# Rename columns to consistent casing if necessary
colnames(shap_bar_data) <- c("Variable", "Mean_SHAP")

shap_bar_plot <- ggplot(shap_bar_data, aes(x = reorder(Variable, Mean_SHAP), y = Mean_SHAP)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    x = "Feature",
    y = "Mean |SHAP value|"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, size = 1),  # add black border
    panel.background = element_blank()  # remove background fill to show border
  )

print(shap_bar_plot)



library(data.table)  # if shap_values is a data.table

# Convert to data.table if not already
shap_dt <- as.data.table(shap_values)

# Filter to only PAD800 and PAD680
shap_dt_sub <- shap_dt[variable %in% c("RIDAGEYR", "LBXTC", "LBXGH", "BPXOSY1")]

# Reset factor levels for variable to only these two
shap_dt_sub[, variable := factor(variable, levels = c("RIDAGEYR", "LBXTC", "LBXGH", "BPXOSY1"))]

# Now plot
shap.plot.summary(shap_dt_sub) + 
  theme(axis.title.y = element_text(face = "bold"))

library(data.table)

# Convert to data.table if not already
shap_dt <- as.data.table(shap_values)

# Rename variables
shap_dt[variable == "RIDAGEYR", variable := "Suraiya"]
shap_dt[variable == "LBXTC", variable := "Kayes"]

# Optionally filter to just those renamed ones
shap_dt_sub <- shap_dt[variable %in% c("Suraiya", "Kayes")]

# Set factor level order (optional for plot order)
shap_dt_sub[, variable := factor(variable, levels = c("Suraiya", "Kayes"))]

# Plot
shap.plot.summary(shap_dt_sub) +
  theme(axis.title.y = element_text(face = "bold"))




library(ggplot2)
library(data.table)

# Assuming shap_dt is your data.table of SHAP values
shap_dt <- as.data.table(shap_values)  # if not already
library(stringr)

# DMDEDUC2
smq_shap <- shap_dt[variable == "DIQ010"]
smq_shap[, rfvalue_cat := as.character(rfvalue)]
smq_shap[rfvalue == 1, rfvalue_cat := "Yes"]
smq_shap[rfvalue == 2, rfvalue_cat := "No"]
smq_shap[rfvalue == 3, rfvalue_cat := "Borderline"]


gg_smq <- ggplot(smq_shap, aes(x = rfvalue_cat, y = value, fill = rfvalue_cat)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  labs(
    x = "Doctor told you have diabetes",
    y = "SHAP value") +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.background = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1)  # Rotates x-axis labels
  )

print(gg_smq)

# SMQ020
smq_shap <- shap_dt[variable == "SMQ020"]
smq_shap[, rfvalue_cat := as.character(rfvalue)]
smq_shap[rfvalue == 1, rfvalue_cat := "Yes"]
smq_shap[rfvalue == 2, rfvalue_cat := "No"]


gg_smq <- ggplot(smq_shap, aes(x = rfvalue_cat, y = value, fill = rfvalue_cat)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  labs(
    x = "Smoked at least 100 cigarettes in life",
    y = "SHAP value") +
  theme_minimal() +
  theme(
    legend.position = "none",
    panel.border = element_rect(color = "black", fill = NA, size = 1),
    panel.background = element_blank(),
    axis.text.x = element_text(angle = 30, hjust = 1)  # Rotates x-axis labels
  )

print(gg_smq)


library(caret)
#prediction
df_test$CVD_risk<-as.factor(df_test$CVD_risk)
df_test$CVD_risk
dtest <- xgb.DMatrix(data = as.matrix( X_test))

xgbPredictions <-predict(fit_xgb, dtest, type = "prob")

print(xgbPredictions)

#write.csv(xgbPredictions, file = "D:\\Research_Work\\Disertation Project 3\\RawData\\FeatureExtraction/CVFS\\probability\\probability_C2E10P0.4.csv", row.names = FALSE)

# 'results' now contains the predicted values


roc_curve <- roc(df_test$CVD_risk, xgbPredictions)
auc_value <- auc(roc_curve)
print(paste("SVM AUC:", auc_value))
# Convert predicted values to class labels
xgbpredicted_labels <- ifelse(xgbPredictions > 0.5, 1, 0)
xgbpredicted_labels
df_test$CVD_risk<-as.factor(df_test$CVD_risk)

summary(df_test$CVD_risk)



xgbpredicted_labels <- factor(xgbpredicted_labels, levels = levels(df_test$CVD_risk))

# Create the confusion matrix
xgbconfusion <- confusionMatrix(xgbpredicted_labels, df_test$CVD_risk, positive ='1', mode = "everything")

# Display the confusion matrix
print(xgbconfusion)
xgbconfusion$byClass

# Assuming 'results' contains your predicted values and 'test' is your test dataset
preds <- xgbpredicted_labels
preds
actuals <- df_test$CVD_risk
actuals

# Calculate MCC
confusion_matrix <- table(Actual = actuals, Predicted = preds)
confusion_matrix
tp <- confusion_matrix["1","1"]  # True Positives
tp
tn <- confusion_matrix["0","0"]  # True Negatives
fp <- confusion_matrix["0","1"]  # False Positives
fn <- confusion_matrix["1","0"]  # False Negatives

mcc <- (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc
# Print MCC

# to rename variables in the shap plot
#y_axis_labels <- attr(shap$variable, "labels")

#y_axis_labels <- unique(shap$variable)
#y_axis_labels


#shap.plot.summary(shap_updated)
draw_confusion_matrix <- function(cm) {
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  
  # Create the matrix
  rect(200, 350, 300, 400, col='#AF97D0')
  rect(300, 350, 400, 400, col='#A7AD50')
  text(250, 405, "Reference", cex=1.3, font=2)
  text(175, 375, "Prediction", cex=1.3, srt=90, font=2)
  
  # Add in the cm results
  res <- as.numeric(cm$table)
  text(250, 375, "-1", cex=1.3, font=2, col='black')
  text(250, 350, "1", cex=1.3, font=2, col='black')
  text(300, 400, "-1", cex=1.3, font=2, col='black')
  text(300, 375, res[1], cex=1.6, font=2, col='black')
  text(300, 350, res[3], cex=1.6, font=2, col='black')
  text(400, 400, "1", cex=1.3, font=2, col='black')
  text(400, 375, res[2], cex=1.6, font=2, col='black')
  text(400, 350, res[4], cex=1.6, font=2, col='black')
}


draw_confusion_matrix <- function(cm) {
  
  layout(matrix(1))
  par(mar=c(2,2,2,2))
  #plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  plot(c(123, 345), c(300, 452), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  #title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#AF97D0')
  text(195, 435, -1, cex=1.2)
  rect(250, 430, 340, 370, col='#A7AD50')
  text(295, 435, 1, cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#A7AD50')
  rect(250, 305, 340, 365, col='#AF97D0')
  text(140, 400, -1, cex=1.2, srt=90)
  text(140, 335, 1, cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='black')
  text(195, 335, res[2], cex=1.6, font=2, col='black')
  text(295, 400, res[3], cex=1.6, font=2, col='black')
  text(295, 335, res[4], cex=1.6, font=2, col='black')
  
  
  
}  
draw_confusion_matrix(xgbconfusion)



