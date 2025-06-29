## t-test
#file<-read.csv(file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/TtestFeatureSet.csv", header=TRUE)
library(dplyr)
##Random forest 
#T-TEST data 
file<-read.csv(file = "D:/Research_Work/research on new idae/idea1/Feature_Extraction/hypergraph/hyp_CDWFS_feat_mapped_d5New.csv", header=TRUE)
#unique(file)
file
set.seed(123)
# Print column names and structure of the data frame
print(colnames(file))
str(file)



selected_data <- file %>%
  ##arrange(original_feature) %>%
  slice_head(n = round(0.25 * nrow(.))) %>%
  select(original_feature)
selected_data

#selected_data<-original_feature

# Assuming your data frame is named selected_data

# Create a new row with the value "Output" in the "original_feature" column
new_row <- data.frame(original_feature = "CVD_risk")

# Add the new row to the last row of the data frame
selected_data <- rbind(selected_data, new_row)

last_column <- selected_data$original_feature
last_column

coln<-colnames(last_column)
coln
len<-length(file)
len

file1<-read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\dataSplit/validate_train_data.csv", header = TRUE)
file1


fil1_back<-file1
file2<-read.csv("D:/Research_Work/research on new idae/idea1/Feature_Extraction\\dataSplit/validate_test_data.csv", header = TRUE)
file2

sub1<-subset(file1,select=last_column)
ncol(sub1)
#sub1["Output"]<-file1$Output
sub2<-subset(file2,select=last_column)
ncol(sub2)
#sub2["Output"]<-file2$Output

######### T-test                  #################
#write.csv(sub1, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_training_merged_file.csv", row.names = FALSE)


#write.csv(sub2, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_validation_merged_file.csv", row.names = FALSE)


###      Random forest             #################
write.csv(sub1, file = "D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\bin5\\data25/selected_training_merged_file.csv", row.names = FALSE)


write.csv(sub2, file = "D:/Research_Work/research on new idae/idea1/Feature_Extraction\\hypergraph\\bin5\\data25/selected_validation_merged_file.csv", row.names = FALSE)


