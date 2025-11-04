## Install rJava package (required for RWeka Java-based classifiers)
## Uncomment the next line if not already installed
# install.packages("rJava")

## Alternative installation from source if above fails
# install.packages('rJava', type = 'source', INSTALL_opts='--merge-multiarch')

## Load necessary R libraries
library(stringr)
library(rattle)  # for data mining tools and Weka interface
library(RWeka)   # provides access to Weka machine learning algorithms
library(rpart)   # recursive partitioning for decision trees
library(ROSE)    # handles class imbalance via resampling
library(dplyr)   # for data manipulation (mutate, filter, etc.)

## Refresh the Weka package manager and install ADTree (Alternating Decision Tree)
WPM("refresh-cache")
WPM("install-package", "alternatingDecisionTrees")

## Load the installed Weka ADTree package
WPM("load-package", "alternatingDecisionTrees")

## Define the Weka classifier path for ADTree and create the classifier
cpath <- "weka/classifiers/trees/ADTree"
ADT <- make_Weka_classifier(cpath)

## Display information about the ADTree classifier
ADT

## Open Weka Options Window for ADTree (optional visualization)
WOW(ADT)

## Read the training dataset (update the file path as needed)
## Ensure the dataset contains a target column named "CVD_risk"
df_trainX <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit\\revised_data\\validate_train_data.csv", header = TRUE)

## Display the number of columns and summary of the target variable
length(df_trainX)
summary(as.factor(df_trainX$CVD_risk))

## (Optional) Handle class imbalance using oversampling or undersampling from ROSE package
# df_trainX <- ovun.sample(Output ~ ., data = df_trainX, method = "over", N=3560, seed=123)$data
# df_trainX <- ovun.sample(Output ~ ., data = df_trainX, method = "under", N=1230, seed=123)$data

## Create a working copy of the dataset
subtrainX <- df_trainX

## Convert target variable to factor (required for classification)
subtrainX$CVD_risk <- as.factor(subtrainX$CVD_risk)

## Convert all character columns to factors for Weka compatibility
subtrainX <- subtrainX %>%
  mutate(across(where(is.character), as.factor))

## Set a random seed for reproducibility
set.seed(123)

## Define a sequence of possible complexity parameter (B) values for ADTree
possiblecValue <- round(seq(from = 5, to = 50, length.out = 55), 0)

## Choose a random sample of values to test
numModels <- 50
cValue <- sample(possiblecValue, numModels)
cValue <- round(cValue)

## Initialize performance metric vectors
pctCorrect <- MAE <- Kappa <- rep(0, numModels)

## Loop through each model and evaluate performance via 10-fold cross-validation
for(i in 1:numModels){
  print(i)
  ## Train ADTree model with current complexity value
  audit.adt <- ADT(CVD_risk ~ ., data = subtrainX, control = Weka_control(B = cValue[i]))
  
  ## Evaluate model using Weka evaluation method
  e <- evaluate_Weka_classifier(audit.adt,
                                numFolds = 10, complexity = TRUE,
                                seed = 123, class = TRUE)
  
  ## Extract and store performance metrics
  evaluation <- e$details
  pctCorrect[i] <- evaluation["pctCorrect"]
  Kappa[i] <- evaluation["kappa"]
  MAE[i] <- evaluation["rootMeanSquaredError"]
}

## Identify the best model (lowest RMSE)
ind <- which.min(MAE)
min(MAE)

## Train final ADTree model using optimal parameter
t <- ADT(CVD_risk ~ ., data = subtrainX, control = Weka_control(B = cValue[ind]))
t

## ---- Visualization Section ----
## Visualize the trained ADTree using Rgraphviz
library("Rgraphviz")

## Save ADTree to a temporary DOT file (GraphViz format)
ff <- tempfile()
write_to_dot(t, ff)

## Save the DOT file to a specific directory (optional)
write_to_dot(t, "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\revised_data\\Fig/su_cord.dot")

## Plot the decision tree structure
plot(agread(ff))
