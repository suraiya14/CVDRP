# Load required library
library(dplyr)

# Step 1: Read the CSV file
df <- read.csv("D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\dataSplit\\validate_train_data.csv",
               header = TRUE, sep = ",")



library(dplyr)




df$CVD_risk[df$CVD_risk == "Yes"]  <- "Y"
df$CVD_risk[df$CVD_risk == "No"]  <- "N"

df$SMQ020[df$SMQ020 == 1]  <- "Yes"
df$SMQ020[df$SMQ020 == 2] <- "No"



df$PAD790U[df$PAD790U == 1]  <- 'D'
df$PAD790U[df$PAD790U == 2]  <- 'W'
df$PAD790U[df$PAD790U == 3]  <- 'M'
df$PAD790U[df$PAD790U == 4]  <- 'Y'

df$PAD790U


df <- df %>%
  mutate(SLD012 = case_when(
    SLD012 == 2 ~ "Less than 3 hours",
    SLD012 == 14 ~ "14 hours or more",
    SLD012 >= 3 & SLD012 < 4 ~ "3 to 3.99",
    SLD012 >= 4 & SLD012 < 5 ~ "4 to 4.99",
    SLD012 >= 5 & SLD012 < 6 ~ "5 to 5.99",
    SLD012 >= 6 & SLD012 < 7 ~ "6 to 6.99",
    SLD012 >= 7 & SLD012 < 8 ~ "7 to 7.99",
    SLD012 >= 8 & SLD012 < 9 ~ "8 to 8.99",
    SLD012 >= 9 & SLD012 < 10 ~ "9 to 9.99",
    SLD012 >= 10 & SLD012 < 11 ~ "10 to 10.99",
    SLD012 >= 11 & SLD012 < 12 ~ "11 to 11.99",
    SLD012 >= 12 & SLD012 < 13 ~ "12 to 12.99",
    SLD012 >= 13 & SLD012 < 14 ~ "13 to 13.99",
    TRUE ~ NA_character_  # For unexpected values
  ))

print(df)


df <- df %>%
  mutate(SLD013 = case_when(
    SLD013 == 2 ~ "Less than 3 hours",
    SLD013 == 14 ~ "14 hours or more",
    SLD013 >= 3 & SLD013 < 4 ~ "3 to 3.99",
    SLD013 >= 4 & SLD013 < 5 ~ "4 to 4.99",
    SLD013 >= 5 & SLD013 < 6 ~ "5 to 5.99",
    SLD013 >= 6 & SLD013 < 7 ~ "6 to 6.99",
    SLD013 >= 7 & SLD013 < 8 ~ "7 to 7.99",
    SLD013 >= 8 & SLD013 < 9 ~ "8 to 8.99",
    SLD013 >= 9 & SLD013 < 10 ~ "9 to 9.99",
    SLD013 >= 10 & SLD013 < 11 ~ "10 to 10.99",
    SLD013 >= 11 & SLD013 < 12 ~ "11 to 11.99",
    SLD013 >= 12 & SLD013 < 13 ~ "12 to 12.99",
    SLD013 >= 13 & SLD013 < 14 ~ "13 to 13.99",
    TRUE ~ NA_character_  # For unexpected values
  ))




df$RIAGENDR[df$RIAGENDR == 1]  <- "Male"
df$RIAGENDR[df$RIAGENDR == 2]  <- "Female"


df$DIQ010[df$DIQ010 == 1]  <- "Yes"
df$DIQ010[df$DIQ010 == 2]  <- "No"
df$DIQ010[df$DIQ010 == 3]  <- "Borderline"


df$RIDRETH3[df$RIDRETH3 == 1]  <- "Mexican American"
df$RIDRETH3[df$RIDRETH3 == 2]  <- "Other Hispanic"
df$RIDRETH3[df$RIDRETH3 == 3]  <- "Non-Hispanic White"
df$RIDRETH3[df$RIDRETH3 == 4]  <- "Non-Hispanic Black"
df$RIDRETH3[df$RIDRETH3 == 6]  <- "	Non-Hispanic Asian"
df$RIDRETH3[df$RIDRETH3 == 7]  <- "Other Race - Including Multi-Racial"


df$DMDEDUC2[df$DMDEDUC2 == 1]  <- "Less than 9th grade"
df$DMDEDUC2[df$DMDEDUC2 == 2]  <- "9-11th grade (Includes 12th grade with no diploma)"
df$DMDEDUC2[df$DMDEDUC2 == 3]  <- "High school graduate/GED or equivalent"
df$DMDEDUC2[df$DMDEDUC2 == 4]  <- "Some college or AA degree"
df$DMDEDUC2[df$DMDEDUC2 == 5]  <- "College graduate or above"

# Optional: Save updated dataframe to new CSV
write.csv(df, "D:\\Research_Work\\research on new idae\\idea1\\Feature_Extraction\\CVFS\\balanced_dataset_CVFS.csv", row.names = FALSE)



