## t-test
#file<-read.csv(file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/TtestFeatureSet.csv", header=TRUE)

##Random forest 
#T-TEST data 
file<-read.csv(file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\ADT_Data.csv", header=TRUE)
#unique(file)
ncol(file)
coln<-colnames(file)
coln
len<-length(file)
len

file1<-read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit\\validate_train_data.csv", header = TRUE)
file1


fil1_back<-file1
file2<-read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit\\validate_test_data.csv", header = TRUE)
#file2_back<-file2

sub1<-subset(file1,select=coln)
ncol(sub1)
#sub1["Output"]<-file1$Output
sub2<-subset(file2,select=coln)
ncol(sub2)
#sub2["Output"]<-file2$Output

######### T-test                  #################
#write.csv(sub1, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_training_merged_file.csv", row.names = FALSE)


#write.csv(sub2, file = "D:\\Research Work\\Disertation Project 1\\Raw Data\\featureExtraction/Tselected_validation_merged_file.csv", row.names = FALSE)


###      Random forest             #################
write.csv(sub1, file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT/selected_training_merged_file.csv", row.names = FALSE)


write.csv(sub2, file = "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\ADT\\selected_validation_merged_file.csv", row.names = FALSE)


