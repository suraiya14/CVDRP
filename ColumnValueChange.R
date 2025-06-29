library(dplyr)

# Step 1: Read the CSV file
df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\data\\balanced_dataset.csv",
               header = TRUE, sep = ",")
df


library(dplyr)




df$Output[df$Output == "Yes"]  <- 1
df$Output[df$Output == "No"]  <- 0

write.csv(df, "D:\\Research_Work\\research on new idae\\idea1\\data\\balanced_dataset.csv", row.names = FALSE)
