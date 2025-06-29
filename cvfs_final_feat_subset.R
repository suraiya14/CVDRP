## t-test
#file<-read.csv(file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/TtestFeatureSet.csv", header=TRUE)
library(dplyr)
##Random forest 
#T-TEST data 
file<-read.csv(file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\C8E10P0.8\\C8E10P0.8New.csv", header=TRUE)
#unique(file)
file
set.seed(123)
# Print column names and structure of the data frame
print(colnames(file))
str(file)
# Assuming 'file' is your data frame
new_row <- data.frame(column_name = "CVD_risk")
new_row
# Use rbind to append the new row
file <- rbind(file, new_row)

# Print the updated data frame
print(file)




# Assuming your data frame is named selected_data

# Add the new row to the last row of the data frame

last_column <- file$column_name
last_column

coln<-colnames(file)
coln
len<-length(file)
len

file1<-read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit/validate_train_data.csv", header = TRUE)
file1


fil1_back<-file1
file2<-read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit/validate_test_data.csv", header = TRUE)
#file2_back<-file2

sub1<-subset(file1,select=last_column)
#sub1<-subset(file1,select=colnames(file))
ncol(sub1)
sub1
#sub1["Output"]<-file1$Output
#sub2<-subset(file2,select=colnames(file))
sub2<-subset(file2,select=last_column)
ncol(sub2)
#sub2["Output"]<-file2$Output

######### T-test                  #################
#write.csv(sub1, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_training_merged_file.csv", row.names = FALSE)


#write.csv(sub2, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_validation_merged_file.csv", row.names = FALSE)


###      Random forest             #################
write.csv(sub1, file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\C8E10P0.8/selected_training_merged_file.csv", row.names = FALSE)


write.csv(sub2, file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\C8E10P0.8/selected_validation_merged_file.csv", row.names = FALSE)


