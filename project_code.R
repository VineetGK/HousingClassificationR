##_____ Import Packages _____##
library(psych)
library(mlbench)
library(caret)
library(randomForest)
library(tidyverse)
library(dplyr)
library(pROC)

##_____ Read Data _____##
data <- read.csv("project_data.csv")

##_____ Data Cleaning _____##
#dimensions
dim(data)

#data types
str(data)

##_____ convert 'class' column to factor _____##
#data$Class <- as.factor(data$Class)
table(data$Class)

##_____ Converting blank spaces into NA's _____##
columns <- colnames(data)
for (col in columns) {
  data[[col]] <- ifelse(trimws(data[[col]]) == "",
                        NA, data[[col]])
}

## Missing values ##
colSums(is.na(data))

## Checking for Dups in Subset dataframe ##
data_dup <- data[!duplicated(data),]
data_dup
dim(data_dup)

## unique values in missing columns ##
unique_vals <- unique(data_dup$CITWP)
unique_vals
##______________________________________________________________________________##
##______________________________________________________________________________##

## Handling Missing Values: Type of data & what to do with it (omit, mean, mode… etc)? ##
#If over ½ (count >= 2159) is N/A then delete

#Remove features w/ NA >= 2159
subset <- subset(data_dup, select = -c(CITWP, ENG, FER, GCM, GCR, JWMNP, JWRIP, MLPA, MLPB, MLPCD, MLPE, MLPFG,
                                       MLPH, MLPIK, MLPJ, SCHG, YOEP, DECADE, DRIVESP, ESP, FOD1P, FOD2P, JWAP,
                                       JWDP, LANP, MIGPUMA, MIGSP, NOP, PAOC, SCIENGP, INDP, POWPUMA, POWSP, RC,
                                       SCIENGRLP, SFN, SFR, VPS))
subset_cols <- colnames(subset)
subset_cols

#fill in 0 for N/A columns
subset <- subset %>% mutate(COW = ifelse(is.na(COW), 0, COW),
                            GCL= ifelse(is.na(GCL), 0, GCL),
                            JWTRNS = ifelse(is.na(JWTRNS), 0, JWTRNS),
                            MARHD = ifelse(is.na(MARHD), 0, MARHD),
                            MARHM = ifelse(is.na(MARHM), 0, MARHM),
                            MARHT = ifelse(is.na(MARHT), 0, MARHT),
                            MARHW = ifelse(is.na(MARHW), 0, MARHW),
                            MARHYP = ifelse(is.na(MARHYP), 0, MARHYP),
                            MIL = ifelse(is.na(MIL), 0, MIL),
                            NWAB = ifelse(is.na(NWAB), 0, NWAB),
                            NWAV = ifelse(is.na(NWAV), 0, NWAV),
                            NWLA = ifelse(is.na(NWLA), 0, NWLA),
                            NWLK = ifelse(is.na(NWLK), 0, NWLK),
                            NWRE = ifelse(is.na(NWRE), 0, NWRE),
                            OCCP = ifelse(is.na(OCCP), 0, OCCP),
                            ESR = ifelse(is.na(ESR), 0, ESR))
#subset

## median for columns w N/A ##
subset <- subset %>% mutate(WKHP = ifelse(is.na(WKHP), median(WKHP, na.rm = TRUE), WKHP),
                            WKWN = ifelse(is.na(WKWN), median(WKWN, na.rm = TRUE), WKWN),
                            PERNP = ifelse(is.na(PERNP), median(PERNP, na.rm = TRUE), PERNP),
                            POVPIP = ifelse(is.na(POVPIP), median(POVPIP, na.rm = TRUE), POVPIP))
#subset

## mode for columns w N/A ##
#function to calculate the mode
get_mode <- function(x) {
  x <- na.omit(x)
  uniq_x <- unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))]
}

## replace NA values with the mode ##
subset <- subset %>% mutate(WRK = replace_na(get_mode(WRK)),)
#subset

## Reclassifying to fill in 3 for N/A in OC (Only Child) column ##
subset <- subset %>% mutate(OC = ifelse(is.na(OC), 2, OC))
#subset

## Reclassifying Class(Prediction value) to 0&1s ##
#subset <- subset %>% mutate(Class = ifelse(Class == "No",0,1))
"unique(subset$Class)

table(subset$Class)
head(subset$Class, 15)"

subset$Class <- as.factor(subset$Class)
contrasts(subset$Class)
unique(subset$Class)
##______________________________________________________________________________##
##______________________________________________________________________________##

##_____ Checking 0 Variance _____##
nearZeroVar(subset, saveMetrics = TRUE)

## remove near 0 var features ##
subset_df <- subset[,nearZeroVar(subset)]
subset
subset_df <- subset %>%
  select(-RT, -DIVISION, -REGION, -STATE, -ADJINC, -WRK, -RACNH)
subset_df

## Remove Y(Class) variable ##
df1 = subset_df[,-c(1, ncol(subset_df))]
df1
cormat = cor(df1)
cormat
high_corr_cols <- findCorrelation(cormat, cutoff = 0.8)

## Print columns to be removed due to perfect correlation (1 or -1) ##
print("Columns to be removed due to perfect correlation (1 or -1):")
print(names(df1)[high_corr_cols])

## Remove the highly correlated columns ##
df_clean <- df1[, -high_corr_cols]

## Print cleaned dataset structure ##
print("Cleaned dataframe:")
str(df_clean)
df_clean

## Add Class back to df_clean ##
df_clean$Class = c(subset_df$Class)
df_clean
table(df_clean$Class)
contrasts(df_clean$Class)

##_____ Data Cleaning Completed _____##
##_____ 0 Variance Check Completed _____## 
##______________________________________________________________________________##
##______________________________________________________________________________##


##_____ PCA _____##
#Define Dataset and Predictor variable
df_clean_x = df_clean[,-c(1, ncol(df_clean))]

x <- df_clean_x
y <- df_clean$Class

str(x)

## Preprocess the data (important for PCA) ##
preprocessed_data <- preProcess(x, method = c("center", "scale")) # Center and scale the data

## Apply preprocessing to the data ##
x_transformed <- predict(preprocessed_data, x)

## Perform PCA ##
pca_result <- prcomp(x_transformed)

## Summary of PCA results ##
summary(pca_result)

## Explained variance plot ##
plot(pca_result, type = "l")

##_____ PCA completed _____##
##______________________________________________________________________________##
##______________________________________________________________________________##



##_____ Outlier Check _____##
#identify outliers
identify_outliers <- function(data) {
  outliers <- list()
  
  for (i in which(sapply(data, is.numeric))) {
    col_name <- names(data)[i]
    
    perc <- quantile(data[,i], c(.25, .75), na.rm=TRUE)
    lower_fence <- perc[1] - 1.5 * IQR(data[,i])
    upper_fence <- perc[2] + 1.5 * IQR(data[,i])
    
    outlier_indices <- which(data[,i] < lower_fence | data[,i] > upper_fence)
    outliers[[col_name]] <- data[outlier_indices, i]
  }
  outliers2 <- outliers[sapply(outliers, length)>0]
  return(outliers2)
}

outliers_vars <-identify_outliers(df_clean)

## Outlier Capping (winsorise extreme values, percent-based capping) ##
pcap2 <- function(df, vars = NULL, percentiles = c(.05, .95)) {
  
  if (is.null(vars)) { 
    # Select both numeric and integer columns
    vars_index <- which(sapply(df, function(x) is.numeric(x) || is.integer(x)))
  } else { 
    vars_index <- which(names(df) %in% vars)  # Use specified variable names
  }
  
  for (i in vars_index) {
    quantiles <- quantile(df[,i], percentiles, na.rm = TRUE)
    lower_bound <- quantiles[1]
    upper_bound <- quantiles[2]
    
    # Replace values below the lower bound and above the upper bound in one step
    df[,i] <- pmax(pmin(df[,i], upper_bound), lower_bound)
  }
  
  return(df)
}


## replace extreme values with percentiles ##
myvars = names(outliers_vars) #column name of outliers
df_clean = pcap2(df_clean, vars = myvars)

## Checking Percentile values of 1st variable ##
quantile(df_clean[,1], c(0.99,1), na.rm = TRUE)

table(df_clean$Class)
dim(df_clean)

##_____ Outlier check completed _____##
##_____ Data Processing Completed _____##

##______________________________________________________________________________##
##______________________________________________________________________________##


##_____ Import Packages _____##
library(ROSE)
library(randomForest)
library(caret)
library(e1071)

dim(df_clean)

##_____ Stratified Data Split: Train & Test _____##
# set seed for reproducibility 
set.seed(123)

## Stratified Sample ##
df_clean <- df_clean %>%
  mutate(row_id = row_number())

## Train Split ##
train_strat <- df_clean %>%
  group_by(Class) %>%
  sample_frac(size = 0.8)
table(train_strat$row_id)

## Test Split ##
test_strat <- df_clean %>%
  anti_join(train_strat, by = "row_id")
table(test_strat$Class)

## Data Splitting Completed ##
##______________________________________________________________________________##
##______________________________________________________________________________##

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____ Class Balancing _____##
set.seed(123)
barplot (prop.table(table(df_clean$Class)), 
         col = rainbow(2),
         ylim = c(0, 1),
         main = "Class Distribution")

# Using ROSE library
## Over sampling ##
str(train_strat)
set.seed(123)

## Train
over_train <- ovun.sample(Class~., data = train_strat, method = "over")$data
table(over_train$Class)
## Test
#over_test <- ovun.sample(Class~., data = test_strat, method = "over")$data
#table(over_test$Class)

## Both (Over & Under) ##
set.seed(123)
## Train
both_train <- ovun.sample(Class~., data = train_strat, method = "both")$data
table(both_train$Class)
## Test
#both_test <- ovun.sample(Class~., data = test_strat, method = "both")$data
#table(both_test$Class)

## Under sampling (not suggested, # of data pts is lower & acc decreases) ##
set.seed(123)
## Train
under_train <- ovun.sample(Class~., data = train_strat, method = "under")$data
table(under_train$Class)
## Test
#under_test <- ovun.sample(Class~., data = test_strat, method = "under")$data
#table(under_test$Class)

## Class Balancing Completed ##
## Will Use 2 Samples: (Over & Both) FROM HERE ON ##
##______________________________________________________________________________##
##______________________________________________________________________________##

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____ Checking for Constant Features _____ ##

check_constant_feature <- function(train) {
  constant_features <- names(train) [sapply(data, function(col) length(unique(col)) == 1)]
  
  if (length(constant_features) >0) {
    cat("Columns with the same value in all rows:\n")
    print(constant_features)
  } else{
    cat("No constant features found.\n")
  }
}

## Constant Features ##
check_constant_feature(over_train)
check_constant_feature(both_train)

## Remove Constant Features ##
over_train <- subset(over_train, select = -c(SPORDER, PWGTP ,  HINS1  , HINS2,  HINS3, row_id))
both_train <- subset(both_train, select = -c(SPORDER, PWGTP ,  HINS1  , HINS2,  HINS3, row_id))
#view(over_train)
#view(both_train)

##______________________________________________________________________________##
##______________________________________________________________________________##

##______________________________________________________________________________##
##___________________ Start Feature Selections & Modeling ______________________##

##_____ Random Forest Feature Selection _____##

##_____ Function to perform Random Forest for feature selection _____##
random_forest_feature_selection <- function(dataset, target_var, importance_threshold = 0.1, seed_value = 123) {
  # Set seed for reproducibility
  set.seed(seed_value)
  
  # Ensure target variable is a factor for classification tasks
  dataset[[target_var]] <- as.factor(dataset[[target_var]])
  
  # Train Random Forest model
  rf_model <- randomForest(as.formula(paste(target_var, "~ .")), data = dataset, importance = TRUE, ntree = 100)
  
  # Get feature importance
  feature_importance <- importance(rf_model)
  
  # Display feature importance
  print("Feature Importance:")
  print(feature_importance)
  
  # Visualize feature importance
  varImpPlot(rf_model)
  
  # Select features with importance greater than the threshold
  important_features <- rownames(feature_importance)[feature_importance[, 1] > importance_threshold]
  
  # Exclude the target variable itself
  important_features <- important_features[important_features != target_var]
  
  # Identify removed features (those with importance lower than the threshold)
  removed_features <- setdiff(colnames(dataset), c(important_features, target_var))
  
  # Create a new dataset with only the important features
  dataset_selected <- dataset[, c(important_features, target_var)]
  
  # Print selected and removed features
  print("Selected Important Features:")
  print(important_features)
  print("Removed Features:")
  print(removed_features)
  
  # Return the new dataset with selected features
  return(dataset_selected)
}

## Apply Random Forest for feature selection on 'over' dataset ##
over_rf_df <- random_forest_feature_selection(over_train, "Class")

## Apply Random Forest for feature selection on 'both' dataset ##
both_rf_df <- random_forest_feature_selection(both_train, "Class")

## Display structure of the cleaned datasets ##
print("Structure of Cleaned 'over' Dataset:")
#str(over_rf_df)

print("Structure of Cleaned 'both' Dataset:")
#str(both_rf_df)

over_rf<- over_rf_df
both_rf<- both_rf_df
test_over_rf <- test_strat
test_both_rf <- test_strat
##______________________________________________________________________________##

##_____  Naive Bayes Model Over RF Features _____##
library(performance)
library(naivebayes)

set.seed(124)

rf_nb_over <- naive_bayes(Class ~ ., data = over_rf_df)
rf_nb_over 
plot(rf_nb_over)

rf_nb_p <- predict(rf_nb_over, test_strat)
rf_nb_cm <- confusionMatrix(rf_nb_p, test_strat$Class)
rf_nb_cm

#TPR No Class
rf_nb_TP <- (rf_nb_cm$table[1,1]) 
rf_nb_FN <- (rf_nb_cm$table[2,1]) 
rf_nb_TP
rf_nb_FN
rf_nb_TPR <- rf_nb_TP / (rf_nb_TP + rf_nb_FN)
print(rf_nb_TPR)
#No TPR/Recall: 0.75

#TNR Yes Class
rf_nb_TN <- (rf_nb_cm$table[2,2]) 
rf_nb_FP <- (rf_nb_cm$table[1,2])
rf_nb_TN
rf_nb_FP
rf_nb_TNR <- (rf_nb_TN / (rf_nb_TN + rf_nb_FP))
print(rf_nb_TNR)
#Yes TPR/Recall: 0.8412698

#FPR No Class
rf_nb_FPR <- rf_nb_FP / (rf_nb_FP + rf_nb_TN)
print(rf_nb_FPR)
#No FPR: 0.1587302

#FPR Yes Class
rf_nb_FPR_2 <- rf_nb_FN / (rf_nb_FN + rf_nb_TP)
print(rf_nb_FPR_2)
#Yes FPR: 0.25

#Precisison No Class
precision <- (rf_nb_TP / (rf_nb_TP + rf_nb_FP))
#Pos Pred Value/Precision No: 0.9836 
recall <- (rf_nb_TN / (rf_nb_TN + rf_nb_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_nb_FP / (rf_nb_FP + rf_nb_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (rf_nb_FN / (rf_nb_FN + rf_nb_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_nb_TP) * as.numeric(rf_nb_TN))
part2 <- (as.numeric(rf_nb_FP) * as.numeric(rf_nb_FN))
part3 <- sqrt((as.numeric(rf_nb_TP) + as.numeric(rf_nb_FP)) * 
                (as.numeric(rf_nb_TP) + as.numeric(rf_nb_FN)) * 
                (as.numeric(rf_nb_TN) + as.numeric(rf_nb_FP)) * 
                (as.numeric(rf_nb_TN) + as.numeric(rf_nb_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_nb_FP) * as.numeric(rf_nb_FN))
part2_2 <- (as.numeric(rf_nb_TP) * as.numeric(rf_nb_TN))
part3_2 <- sqrt((as.numeric(rf_nb_FP) + as.numeric(rf_nb_TP)) * 
                  (as.numeric(rf_nb_FP) + as.numeric(rf_nb_TN)) * 
                  (as.numeric(rf_nb_FN) + as.numeric(rf_nb_TP)) * 
                  (as.numeric(rf_nb_FN) + as.numeric(rf_nb_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_nb_prob <- predict(rf_nb_over, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_strat$Class, rf_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for RF Naive Bayes Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9836
#Neg Pred Value/Precision Yes: 0.2095
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3378904
#MCC Yes: -0.3378904
#ROC: 0.8729
##______________________________________________________________________________##

##_____  Naive Bayes Model Both RF Features _____##
set.seed(125)

rf_nb_both <- naive_bayes(Class ~ ., data = both_rf_df)
rf_nb_both 
plot(rf_nb_both)

rf_nb_p_2 <- predict(rf_nb_both, test_strat)
rf_nb_cm_2 <- confusionMatrix(rf_nb_p_2, test_strat$Class)
rf_nb_cm_2

#TPR No Class
rf_nb_TP_2 <- (rf_nb_cm_2$table[1,1]) 
rf_nb_FN_2 <- (rf_nb_cm_2$table[2,1]) 
rf_nb_TP_2
rf_nb_FN_2
rf_nb_TPR_2 <- rf_nb_TP_2 / (rf_nb_TP_2 + rf_nb_FN_2)
print(rf_nb_TPR_2)
#No TPR/Recall: 0.75

#TNR Yes Class
rf_nb_TN_2 <- (rf_nb_cm_2$table[2,2]) 
rf_nb_FP_2 <- (rf_nb_cm_2$table[1,2])
rf_nb_TN_2
rf_nb_FP_2
rf_nb_TNR_2 <- (rf_nb_TN_2 / (rf_nb_TN_2 + rf_nb_FP_2))
print(rf_nb_TNR_2)
#Yes TPR/Recall: 0.8412698

#FPR No Class
rf_nb_FPR_2 <- rf_nb_FP_2 / (rf_nb_FP_2 + rf_nb_TN_2)
print(rf_nb_FPR_2)
#No FPR: 0.1587302

#FPR Yes Class
rf_nb_FPR_2 <- rf_nb_FN_2 / (rf_nb_FN_2 + rf_nb_TP_2)
print(rf_nb_FPR_2)
#Yes FPR: 0.25

#Precisison No Class
precision_2 <- (rf_nb_TP_2 / (rf_nb_TP_2 + rf_nb_FP_2))
#Pos Pred Value/Precision No: 0.9836 
recall_2 <- (rf_nb_TN_2 / (rf_nb_TN_2 + rf_nb_FP_2))

F1 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_nb_FP_2 / (rf_nb_FP_2 + rf_nb_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.1978
recall_2 <- (rf_nb_FN_2 / (rf_nb_FN_2 + rf_nb_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_nb_TP_2) * as.numeric(rf_nb_TN_2))
part2 <- (as.numeric(rf_nb_FP_2) * as.numeric(rf_nb_FN_2))
part3 <- sqrt((as.numeric(rf_nb_TP_2) + as.numeric(rf_nb_FP_2)) * 
                (as.numeric(rf_nb_TP_2) + as.numeric(rf_nb_FN_2)) * 
                (as.numeric(rf_nb_TN_2) + as.numeric(rf_nb_FP_2)) * 
                (as.numeric(rf_nb_TN_2) + as.numeric(rf_nb_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_nb_FP_2) * as.numeric(rf_nb_FN_2))
part2_2 <- (as.numeric(rf_nb_TP_2) * as.numeric(rf_nb_TN_2))
part3_2 <- sqrt((as.numeric(rf_nb_FP_2) + as.numeric(rf_nb_TP_2)) * 
                  (as.numeric(rf_nb_FP_2) + as.numeric(rf_nb_TN_2)) * 
                  (as.numeric(rf_nb_FN_2) + as.numeric(rf_nb_TP_2)) * 
                  (as.numeric(rf_nb_FN_2) + as.numeric(rf_nb_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_nb_prob <- predict(rf_nb_both, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc_both <- roc(test_strat$Class, rf_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc_both, main = "ROC Curve for RF Naive Bayes Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc_both <- auc(roc_both)
print(roc_auc_both)
#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Neg Pred Value/Precision Yes: 0.1978
#Pos Pred Value/Precision No: 0.9836
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3218696
#MCC Yes: -0.3218696
#ROC: 0.8748
##______________________________________________________________________________##

##_____  Weighted RF Model Over RF Features _____##
set.seed(226)
wrf_train_over <- randomForest(Class ~., data = over_rf_df, sampsize = c("Yes" = 185, "No" = 140), importance = TRUE)
wrf_train_over
rf_wrf_p <- predict(wrf_train_over, test_strat)
rf_wrf_cm<- confusionMatrix(rf_wrf_p, test_strat$Class)
rf_wrf_cm

#TPR No Class
rf_wrf_TP <- (rf_wrf_cm$table[1,1]) 
rf_wrf_FN<- (rf_wrf_cm$table[2,1]) 
rf_wrf_TP
rf_wrf_FN
rf_wrf_TPR <- rf_wrf_TP / (rf_wrf_TP + rf_wrf_FN)
print(rf_wrf_TPR)
#TPR: 0.815

#TNR Yes Class
rf_wrf_TN <- (rf_wrf_cm$table[2,2]) 
rf_wrf_FP <- (rf_wrf_cm$table[1,2])
rf_wrf_TN
rf_wrf_FP
rf_wrf_TNR <- (rf_wrf_TN / (rf_wrf_TN + rf_wrf_FP))
print(rf_wrf_TNR)
#TPR: 0.8095238

#FPR No Class
rf_wrf_FPR <- rf_wrf_FP / (rf_wrf_FP + rf_wrf_TN)
print(rf_wrf_FPR)
#No FPR: 0.1904762

#FPR Yes Class
rf_wrf_FPR_2 <- rf_wrf_FN / (rf_wrf_FN + rf_wrf_TP)
print(rf_wrf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (rf_wrf_TP / (rf_wrf_TP + rf_wrf_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (rf_wrf_TN / (rf_wrf_TN + rf_wrf_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_wrf_FP / (rf_wrf_FP + rf_wrf_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (rf_wrf_FN / (rf_wrf_FN + rf_wrf_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_wrf_TP) * as.numeric(rf_wrf_TN))
part2 <- (as.numeric(rf_wrf_FP) * as.numeric(rf_wrf_FN))
part3 <- sqrt((as.numeric(rf_wrf_TP) + as.numeric(rf_wrf_FP)) * 
                (as.numeric(rf_wrf_TP) + as.numeric(rf_wrf_FN)) * 
                (as.numeric(rf_wrf_TN) + as.numeric(rf_wrf_FP)) * 
                (as.numeric(rf_wrf_TN) + as.numeric(rf_wrf_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_wrf_FP) * as.numeric(rf_wrf_FN))
part2_2 <- (as.numeric(rf_wrf_TP) * as.numeric(rf_wrf_TN))
part3_2 <- sqrt((as.numeric(rf_wrf_FP) + as.numeric(rf_wrf_TP)) * 
                  (as.numeric(rf_wrf_FP) + as.numeric(rf_wrf_TN)) * 
                  (as.numeric(rf_wrf_FN) + as.numeric(rf_wrf_TP)) * 
                  (as.numeric(rf_wrf_FN) + as.numeric(rf_wrf_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_wrf_prob <- predict(wrf_train_over, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_strat$Class, rf_wrf_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for Random Forest Weighted RF Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.8150
#Yes TPR/Recall: 0.8095
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9819
#Neg Pred Value/Precision Yes: 0.2563
#F1 No: 0.8874299
#F1 Yes: 0.03292791
#Kappa : 0.2475
#MCC No: 0.3857036
#MCC Yes: -0.3857036
#ROC: 0.8659

#boruta_over <- Boruta(Class ~., data = over_rf_df, doTrace = 3, maxRuns = 250)
#print(boruta_over)

#plot(boruta_over, las=2, cex.axis =0.7)
#plotImpHistory(boruta_over)
##______________________________________________________________________________##

##_____  Weighted RF Model Both RF Features _____##
set.seed(127)
wrf_train_both <- randomForest(Class ~., data = both_rf_df, sampsize = c("Yes" = 185, "No" = 140), importance = TRUE)
wrf_train_both
rf_wrf_p_2 <- predict(wrf_train_both, test_strat)
rf_wrf_cm_2 <- confusionMatrix(rf_wrf_p_2, test_strat$Class)
rf_wrf_cm_2

#TPR No Class
rf_wrf_TP_2 <- (rf_wrf_cm_2$table[1,1]) 
rf_wrf_FN_2 <- (rf_wrf_cm_2$table[2,1]) 
rf_wrf_TP_2
rf_wrf_FN_2
rf_wrf_TPR_2 <- rf_wrf_TP_2 / (rf_wrf_TP_2 + rf_wrf_FN_2)
print(rf_wrf_TPR_2)
#TPR: 0.795

#TNR Yes Class
rf_wrf_TN_2 <- (rf_wrf_cm_2$table[2,2]) 
rf_wrf_FP_2 <- (rf_wrf_cm_2$table[1,2])
rf_wrf_TN_2
rf_wrf_FP_2
rf_wrf_TNR_2 <- (rf_wrf_TN_2 / (rf_wrf_TN_2 + rf_wrf_FP_2))
print(rf_wrf_TNR_2)
#TPR: 0.8095238

#FPR No Class
rf_wrf_FPR_2 <- rf_wrf_FP_2 / (rf_wrf_FP_2 + rf_wrf_TN_2)
print(rf_wrf_FPR_2)
#No FPR: 0.1904762

#FPR Yes Class
rf_wrf_FPR_2 <- rf_wrf_FN_2 / (rf_wrf_FN_2 + rf_wrf_TP_2)
print(rf_wrf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (rf_wrf_TP_2 / (rf_wrf_TP_2 + rf_wrf_FP_2))
#Pos Pred Value/Precision No: 0.8874299
recall <- (rf_wrf_TN_2 / (rf_wrf_TN_2 + rf_wrf_FP_2))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_wrf_FP_2 / (rf_wrf_FP_2 + rf_wrf_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (rf_wrf_FN_2 / (rf_wrf_FN_2 + rf_wrf_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_wrf_TP_2) * as.numeric(rf_wrf_TN_2))
part2 <- (as.numeric(rf_wrf_FP_2) * as.numeric(rf_wrf_FN_2))
part3 <- sqrt((as.numeric(rf_wrf_TP_2) + as.numeric(rf_wrf_FP_2)) * 
                (as.numeric(rf_wrf_TP_2) + as.numeric(rf_wrf_FN_2)) * 
                (as.numeric(rf_wrf_TN_2) + as.numeric(rf_wrf_FP_2)) * 
                (as.numeric(rf_wrf_TN_2) + as.numeric(rf_wrf_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_wrf_FP_2) * as.numeric(rf_wrf_FN_2))
part2_2 <- (as.numeric(rf_wrf_TP_2) * as.numeric(rf_wrf_TN_2))
part3_2 <- sqrt((as.numeric(rf_wrf_FP_2) + as.numeric(rf_wrf_TP_2)) * 
                  (as.numeric(rf_wrf_FP_2) + as.numeric(rf_wrf_TN_2)) * 
                  (as.numeric(rf_wrf_FN_2) + as.numeric(rf_wrf_TP_2)) * 
                  (as.numeric(rf_wrf_FN_2) + as.numeric(rf_wrf_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_wrf_prob_2 <- predict(wrf_train_both, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_strat$Class, rf_wrf_prob_2[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for Random Forest Weighted RF Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.8666

#boruta_both <- Boruta(Class ~., data = both_rf_df, doTrace = 3, maxRuns = 250)
#print(boruta_both)

#plot(boruta_both, las=2, cex.axis =0.7)
#plotImpHistory(boruta_both)
##______________________________________________________________________________##

##_____  NN Model On RF Over Features _____##
library(NeuralNetTools)
library(nnet)
set.seed(128)

over_rf_df <- random_forest_feature_selection(over_train, "Class")
over_rf_df$Class <- as.factor(over_rf_df$Class)

# min-max normalization to [0, 1]
over_rf_scaled <- over_rf_df %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))
head(over_rf_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
rf_over_features <- colnames(over_rf_df)
rf_over_features
nn_test_rf <- test_strat %>% select(all_of(rf_over_features))
nn_test_rf_scaled <- nn_test_rf %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(over_rf_scaled)
table(over_rf_scaled$Class)
levels(nn_test_rf_scaled)
levels(nn_test_rf_scaled$Class)
# build model with three units in hidden layer
nnet_over_model <- nnet(Class ~ ., data = over_rf_scaled, size = 5, trace = TRUE)
nnet_over_model
plotnet(nnet_over_model)

# test the model
pred <- predict(nnet_over_model, newdata = nn_test_rf_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_rf_scaled$Class))

levels(pred)
levels(nn_test_rf_scaled$Class)
rf_nn_cm <- confusionMatrix(pred, nn_test_rf_scaled$Class)
rf_nn_cm
#Accuracy : 0.8772
#Sensitivity : 0.9012        
#Specificity : 0.5714 
#Kappa : 0.3425 
#FPR No Class
#TPR No Class
rf_nn_TP_2 <- (rf_nn_cm$table[1,1]) 
rf_nn_FN_2 <- (rf_nn_cm$table[2,1]) 
rf_nn_TP_2
rf_nn_FN_2
rf_nn_TPR_2 <- rf_nn_TP_2 / (rf_nn_TP_2 + rf_nn_FN_2)
print(rf_nn_TPR_2)
#TPR: 0.795

#TNR Yes Class
rf_nn_TN_2 <- (rf_nn_cm$table[2,2]) 
rf_nn_FP_2 <- (rf_nn_cm$table[1,2])
rf_nn_TN_2
rf_nn_FP_2
rf_nn_TNR_2 <- (rf_nn_TN_2 / (rf_nn_TN_2 + rf_nn_FP_2))
print(rf_nn_TNR_2)
#TPR: 0.8095238

#FPR No Class
rf_nn_FPR_2 <- rf_nn_FP_2 / (rf_nn_FP_2 + rf_nn_TN_2)
print(rf_nn_FPR_2)
#No FPR: 0.1904762

#FPR Yes Class
rf_nn_FPR_2 <- rf_nn_FN_2 / (rf_nn_FN_2 + rf_nn_TP_2)
print(rf_nn_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (rf_nn_TP_2 / (rf_nn_TP_2 + rf_nn_FP_2))
#Pos Pred Value/Precision No: 0.8874299
recall <- (rf_nn_TN_2 / (rf_nn_TN_2 + rf_nn_FP_2))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_nn_FP_2 / (rf_nn_FP_2 + rf_nn_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (rf_nn_FN_2 / (rf_nn_FN_2 + rf_nn_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_nn_TP_2) * as.numeric(rf_nn_TN_2))
part2 <- (as.numeric(rf_nn_FP_2) * as.numeric(rf_nn_FN_2))
part3 <- sqrt((as.numeric(rf_nn_TP_2) + as.numeric(rf_nn_FP_2)) * 
                (as.numeric(rf_nn_TP_2) + as.numeric(rf_nn_FN_2)) * 
                (as.numeric(rf_nn_TN_2) + as.numeric(rf_nn_FP_2)) * 
                (as.numeric(rf_nn_TN_2) + as.numeric(rf_nn_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_nn_FP_2) * as.numeric(rf_nn_FN_2))
part2_2 <- (as.numeric(rf_nn_TP_2) * as.numeric(rf_nn_TN_2))
part3_2 <- sqrt((as.numeric(rf_nn_FP_2) + as.numeric(rf_nn_TP_2)) * 
                  (as.numeric(rf_nn_FP_2) + as.numeric(rf_nn_TN_2)) * 
                  (as.numeric(rf_nn_FN_2) + as.numeric(rf_nn_TP_2)) * 
                  (as.numeric(rf_nn_FN_2) + as.numeric(rf_nn_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_nn_prob <- predict(nnet_over_model, nn_test_rf_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_rf_scaled$Class, rf_nn_prob)  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for Random Forest Neural Net Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.8234


##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

#_____ Parameter tuning RF Over Features ______
#set.seed(129)
#ctrl <- trainControl(method = "CV", number = 10,
#summaryFunction = twoClassSummary,
#classProbs = TRUE,
#savePredictions = TRUE)

#nnetGrid <- expand.grid(size = 1:13, decay = seq(0, 2, 0.2))

#set.seed(130)
#nnetFit <- train(x = over_rf_scaled[, -36], 
#y = over_rf_scaled$Class,
#method = "nnet",
#metric = "ROC",
#preProc = c("center", "scale"),
#tuneGrid = nnetGrid,
#trace = FALSE,
#maxit = 100,
#MaxNWts = 1000,
#trControl = ctrl)
#nnetFit
#nnetFit$bestTune
#plot(nnetFit)

#plotnet(nnetFit$finalModel)

#test_pred <- predict(nnetFit, newdata = nn_test_rf_scaled)
#test_pred

#performance_measures <- confusionMatrix(test_pred, nn_test_rf_scaled$Class)
#performance_measures"
#Accuracy : 0.8702
#Sensitivity : 0.9012         
#Specificity : 0.4762
#Kappa : 0.2824
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

##______________________________________________________________________________##

#_____  NN Model On RF Both Features _____
set.seed(128)

both_rf_df <- random_forest_feature_selection(both_train, "Class")
both_rf_df$Class <- as.factor(both_rf_df$Class)

# min-max normalization to [0, 1]
both_rf_scaled <- both_rf_df %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))
head(both_rf_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
rf_both_features <- colnames(both_rf_df)
rf_both_features
nn_test_rf <- test_strat %>% select(all_of(rf_both_features))
nn_test_rf_scaled <- nn_test_rf %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(both_rf_scaled)
table(both_rf_scaled$Class)
levels(nn_test_rf_scaled)
levels(nn_test_rf_scaled$Class)
# build model with three units in hidden layer
nnet_both_model <- nnet(Class ~ ., data = both_rf_scaled, size = 5, trace = TRUE)
nnet_both_model
plotnet(nnet_both_model)

# test the model
pred <- predict(nnet_both_model, newdata = nn_test_rf_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_rf_scaled$Class))

levels(pred)
levels(nn_test_rf_scaled$Class)
rf_nn_cm <- confusionMatrix(pred, nn_test_rf_scaled$Class)
rf_nn_cm
#Accuracy : 0.8772
#Sensitivity : 0.9012        
#Specificity : 0.5714 
#Kappa : 0.3425 
#FPR No Class
#TPR No Class
rf_nn_TP_2 <- (rf_nn_cm$table[1,1]) 
rf_nn_FN_2 <- (rf_nn_cm$table[2,1]) 
rf_nn_TP_2
rf_nn_FN_2
rf_nn_TPR_2 <- rf_nn_TP_2 / (rf_nn_TP_2 + rf_nn_FN_2)
print(rf_nn_TPR_2)
#TPR: 0.795

#TNR Yes Class
rf_nn_TN_2 <- (rf_nn_cm$table[2,2]) 
rf_nn_FP_2 <- (rf_nn_cm$table[1,2])
rf_nn_TN_2
rf_nn_FP_2
rf_nn_TNR_2 <- (rf_nn_TN_2 / (rf_nn_TN_2 + rf_nn_FP_2))
print(rf_nn_TNR_2)
#TPR: 0.8095238

#FPR No Class
rf_nn_FPR_2 <- rf_nn_FP_2 / (rf_nn_FP_2 + rf_nn_TN_2)
print(rf_nn_FPR_2)
#No FPR: 0.1904762

#FPR Yes Class
rf_nn_FPR_2 <- rf_nn_FN_2 / (rf_nn_FN_2 + rf_nn_TP_2)
print(rf_nn_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (rf_nn_TP_2 / (rf_nn_TP_2 + rf_nn_FP_2))
#Pos Pred Value/Precision No: 0.8874299
recall <- (rf_nn_TN_2 / (rf_nn_TN_2 + rf_nn_FP_2))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (rf_nn_FP_2 / (rf_nn_FP_2 + rf_nn_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (rf_nn_FN_2 / (rf_nn_FN_2 + rf_nn_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(rf_nn_TP_2) * as.numeric(rf_nn_TN_2))
part2 <- (as.numeric(rf_nn_FP_2) * as.numeric(rf_nn_FN_2))
part3 <- sqrt((as.numeric(rf_nn_TP_2) + as.numeric(rf_nn_FP_2)) * 
                (as.numeric(rf_nn_TP_2) + as.numeric(rf_nn_FN_2)) * 
                (as.numeric(rf_nn_TN_2) + as.numeric(rf_nn_FP_2)) * 
                (as.numeric(rf_nn_TN_2) + as.numeric(rf_nn_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(rf_nn_FP_2) * as.numeric(rf_nn_FN_2))
part2_2 <- (as.numeric(rf_nn_TP_2) * as.numeric(rf_nn_TN_2))
part3_2 <- sqrt((as.numeric(rf_nn_FP_2) + as.numeric(rf_nn_TP_2)) * 
                  (as.numeric(rf_nn_FP_2) + as.numeric(rf_nn_TN_2)) * 
                  (as.numeric(rf_nn_FN_2) + as.numeric(rf_nn_TP_2)) * 
                  (as.numeric(rf_nn_FN_2) + as.numeric(rf_nn_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
rf_nn_prob_2 <- predict(nnet_both_model, nn_test_rf_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_rf_scaled$Class, rf_nn_prob_2[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for Random Forest Neural Net Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.8666


##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

## Parameter tuning ##
#set.seed(132)
#ctrl_2 <- trainControl(method = "CV", number = 10,
#                     summaryFunction = twoClassSummary,
#                     classProbs = TRUE,
#                     savePredictions = TRUE)
#
#nnetGrid_2 <- expand.grid(size = 1:13, decay = seq(0, 2, 0.2))

#set.seed(223)
#nnetFit_2 <- train(x = both_rf_scaled[, -41], 
#                 y = both_rf_scaled$Class,
#                 method = "nnet",
#                 metric = "ROC",
#                 preProc = c("center", "scale"),
#                tuneGrid = nnetGrid_2,
#                 trace = FALSE,
#                 maxit = 100,
#                 MaxNWts = 1000,
#                 trControl = ctrl_2)
#nnetFit_2
#nnetFit_2$bestTune
#plot(nnetFit_2)

#plotnet(nnetFit_2$finalModel)

#test_pred_2 <- predict(nnetFit_2, newdata = nn_test_rf_2_scaled)
#test_pred_2
#test_strat$Class

#performance_measures_2 <- confusionMatrix(test_pred_2, nn_test_rf_2_scaled$Class)
#performance_measures_2

#Accuracy : 0.8702
#Sensitivity : 0.9012         
#Specificity : 0.4762
#Kappa : 0.2824

##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

#logistic-------------------------------------
# Load necessary libraries
library(caret)
library(pROC)

# Define the logistic regression function with evaluation metrics
logistic_regression_evaluation <- function(train_data, test_data, target_class) {
  
  # Ensure the target variable is a factor
  train_data[[target_class]] <- as.factor(train_data[[target_class]])
  test_data[[target_class]] <- as.factor(test_data[[target_class]])
  
  # Define the logistic regression model formula
  model_formula <- as.formula(paste(target_class, "~ ."))
  
  # Train the logistic regression model
  logistic_model <- glm(model_formula, data = train_data, family = "binomial")
  
  # Predict probabilities on the test set
  pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
  
  # Convert probabilities to class predictions (threshold = 0.5)
  pred_class <- ifelse(pred_prob > 0.5, levels(train_data[[target_class]])[2], 
                       levels(train_data[[target_class]])[1])
  
  # Convert to factor for evaluation
  pred_class <- factor(pred_class, levels = levels(test_data[[target_class]]))
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(pred_class, test_data[[target_class]])
  
  # Extract performance metrics
  precision   <- conf_matrix$byClass["Pos Pred Value"]
  recall      <- conf_matrix$byClass["Sensitivity"]
  f_measure   <- 2 * ((precision * recall) / (precision + recall))
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  kappa_value <- conf_matrix$overall["Kappa"]
  
  # Compute ROC and AUC (set positive class explicitly)
  roc_curve <- roc(test_data[[target_class]], pred_prob, 
                   levels = levels(test_data[[target_class]]), 
                   direction = "<")  # Ensure correct ordering
  auc_value <- auc(roc_curve)
  
  # Compute Matthews Correlation Coefficient (MCC) with numeric conversion to prevent overflow
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[1, 2])
  FN <- as.numeric(conf_matrix$table[2, 1])
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc_value <- ifelse(denominator == 0, 0, numerator / denominator)  # Avoid division by zero
  
  # Return a list with all the performance metrics
  return(list(
    Precision   = precision,
    Recall      = recall,
    F_Measure   = f_measure,
    ROC_AUC     = auc_value,
    MCC         = mcc_value,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Kappa       = kappa_value
  ))
}






result <- logistic_regression_evaluation(over_rf, test_over_rf, "Class")
print(result)
result1 <- logistic_regression_evaluation(both_rf, test_both_rf, "Class")
print(result1)
#decision tree---------------------------------------------

# Load required libraries
library(caret)
library(pROC)
library(e1071)
library(rpart)

# Define the function to perform Decision Tree and calculate evaluation metrics
decision_tree_evaluation <- function(train_data, test_data, target_class) {
  
  # Ensure the target variable is a factor
  train_data[[target_class]] <- as.factor(train_data[[target_class]])
  test_data[[target_class]] <- as.factor(test_data[[target_class]])
  
  # Ensure no missing values in the target variable
  if (any(is.na(train_data[[target_class]])) | any(is.na(test_data[[target_class]]))) {
    stop("Target variable contains missing values. Please handle them before proceeding.")
  }
  
  # Define the decision tree model formula
  model_formula <- as.formula(paste(target_class, "~ ."))
  
  # Train the decision tree model using rpart
  decision_tree_model <- rpart(model_formula, data = train_data, method = "class")
  
  # Predict class probabilities on the test set
  pred_prob <- predict(decision_tree_model, newdata = test_data, type = "prob")
  
  # Convert predicted probabilities to class predictions (threshold = 0.5)
  pred_class <- ifelse(pred_prob[,2] > 0.5, "Yes", "No")
  
  # Convert predictions to factor with correct levels
  pred_class <- factor(pred_class, levels = levels(test_data[[target_class]]))
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(pred_class, test_data[[target_class]])
  
  # Extract performance metrics
  precision   <- conf_matrix$byClass["Pos Pred Value"]
  recall      <- conf_matrix$byClass["Sensitivity"]
  f_measure   <- 2 * ((precision * recall) / (precision + recall))
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  kappa_value <- conf_matrix$overall["Kappa"]
  
  # Compute ROC and AUC
  roc_curve <- roc(test_data[[target_class]], pred_prob[,2], levels = rev(levels(test_data[[target_class]])), direction = "<")
  auc_value <- auc(roc_curve)
  
  # Compute Matthews Correlation Coefficient (MCC)
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[1, 2])
  FN <- as.numeric(conf_matrix$table[2, 1])
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc_value <- ifelse(denominator == 0, 0, numerator / denominator)
  
  # Return a list with all the performance metrics
  return(list(
    Precision   = precision,
    Recall      = recall,
    F_Measure   = f_measure,
    ROC_AUC     = auc_value,
    MCC         = mcc_value,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Kappa       = kappa_value
  ))
}

# Example Usage:
# Assuming you have a dataset already split into train_data and test_data, and the target variable is named "target_variable".
result <- decision_tree_evaluation(over_rf, test_over_rf, "Class")
print(result)

result1 <- decision_tree_evaluation(both_rf, test_both_rf, "Class")
print(result)


#svm------------------------------------------------

# Load required libraries
library(caret)
library(pROC)
library(e1071)

# Define the function to perform SVM and calculate evaluation metrics
svm_evaluation <- function(train_data, test_data, target_class) {
  
  # Ensure the target variable is a factor
  train_data[[target_class]] <- as.factor(train_data[[target_class]])
  test_data[[target_class]] <- as.factor(test_data[[target_class]])
  
  # Ensure no missing values in the target variable
  if (any(is.na(train_data[[target_class]])) | any(is.na(test_data[[target_class]]))) {
    stop("Target variable contains missing values. Please handle them before proceeding.")
  }
  
  # Define the SVM model formula
  model_formula <- as.formula(paste(target_class, "~ ."))
  
  # Train the SVM model using caret with the method 'svmRadial' for SVM with radial basis function kernel
  svm_model <- train(model_formula, data = train_data, method = "svmRadial", 
                     trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
  
  # Predict probabilities on the test set
  pred_prob <- predict(svm_model, newdata = test_data, type = "prob")
  
  # Convert predicted probabilities to class predictions (threshold = 0.5)
  pred_class <- ifelse(pred_prob[,2] > 0.5, "Yes", "No")
  
  # Convert predictions to factor with correct levels
  pred_class <- factor(pred_class, levels = levels(test_data[[target_class]]))
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(pred_class, test_data[[target_class]])
  
  # Extract performance metrics
  precision   <- conf_matrix$byClass["Pos Pred Value"]
  recall      <- conf_matrix$byClass["Sensitivity"]
  f_measure   <- 2 * ((precision * recall) / (precision + recall))
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  kappa_value <- conf_matrix$overall["Kappa"]
  
  # Compute ROC and AUC
  roc_curve <- roc(test_data[[target_class]], pred_prob[,2], levels = rev(levels(test_data[[target_class]])), direction = "<")
  auc_value <- auc(roc_curve)
  
  # Compute Matthews Correlation Coefficient (MCC)
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[1, 2])
  FN <- as.numeric(conf_matrix$table[2, 1])
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc_value <- ifelse(denominator == 0, 0, numerator / denominator)
  
  # Return a list with all the performance metrics
  return(list(
    Precision   = precision,
    Recall      = recall,
    F_Measure   = f_measure,
    ROC_AUC     = auc_value,
    MCC         = mcc_value,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Kappa       = kappa_value
  ))
}


resultsvm <- svm_evaluation(over_rf, test_over_rf, "Class")
print(resultsvm)

resultsvm1 <- svm_evaluation(both_rf, test_both_rf, "Class")
print(resultsvm1)




##______________________________________________________________________________##
##______________________________________________________________________________##

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____ Near Zero Variance Feature Selection _____##
## Function to remove near-zero variance features ##
remove_nzv_features <- function(dataset) {
  # Identify near-zero variance features
  nzv_features <- nearZeroVar(dataset, saveMetrics = TRUE)
  
  # Get names of features to remove
  removed_features <- rownames(nzv_features[nzv_features$nzv, ])
  
  # Remove NZV features from dataset
  dataset_cleaned <- dataset[, !names(dataset) %in% removed_features]
  
  # Print removed features
  print("Removed Near-Zero Variance Features:")
  print(removed_features)
  
  return(dataset_cleaned)
}

# Apply function to both datasets
over_train_nzv <- remove_nzv_features(over_train)
both_train_nzv <- remove_nzv_features(both_train)

# Check the structure of cleaned datasets
print("Structure of 'over' dataset after NZV removal:")
str(over_train_nzv)

print("Structure of 'both' dataset after NZV removal:")
str(both_train_nzv)

over_nzv <- over_train_nzv 
both_nzv <- both_train_nzv
test_over_nzv <- test_strat
test_both_nzv <- test_strat
##______________________________________________________________________________##
##_____  Naive Bayes Model Over NZV Features _____##
set.seed(140)

nzv_nb_over <- naive_bayes(Class ~ ., data = over_train_nzv)
nzv_nb_over 
plot(nzv_nb_over)

nzv_nb_p <- predict(nzv_nb_over, test_strat)
nzv_nb_cm <- confusionMatrix(nzv_nb_p, test_strat$Class)
nzv_nb_cm

#TPR No Class
nzv_nb_TP <- (nzv_nb_cm$table[1,1]) 
nzv_nb_FN <- (nzv_nb_cm$table[2,1]) 
nzv_nb_TP
nzv_nb_FN
nzv_nb_TPR <- nzv_nb_TP / (nzv_nb_TP + nzv_nb_FN)
print(nzv_nb_TPR)
#TPR: 0.75

#TNR Yes Class
nzv_nb_TN <- (nzv_nb_cm$table[2,2]) 
nzv_nb_FP <- (nzv_nb_cm$table[1,2])
nzv_nb_TN
nzv_nb_FP
nzv_nb_TNR <- (nzv_nb_TN / (nzv_nb_TN + nzv_nb_FP))
print(nzv_nb_TNR)
#TPR: 0.8412698

#FPR No Class
nzv_nb_FPR <- nzv_nb_FP / (nzv_nb_FP + nzv_nb_TN)
print(nzv_nb_FPR)
#No FPR: 0.1587302

#FPR Yes Class
nzv_nb_FPR_2 <- nzv_nb_FN / (nzv_nb_FN + nzv_nb_TP)
print(nzv_nb_FPR_2)
#Yes FPR: 0.25

#Precisison No Class
precision <- (nzv_nb_TP / (nzv_nb_TP + nzv_nb_FP))
#Pos Pred Value/Precision No: 0.9836 
recall <- (nzv_nb_TN / (nzv_nb_TN + nzv_nb_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_nb_FP / (nzv_nb_FP + nzv_nb_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (nzv_nb_FN / (nzv_nb_FN + nzv_nb_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_nb_TP) * as.numeric(nzv_nb_TN))
part2 <- (as.numeric(nzv_nb_FP) * as.numeric(nzv_nb_FN))
part3 <- sqrt((as.numeric(nzv_nb_TP) + as.numeric(nzv_nb_FP)) * 
                (as.numeric(nzv_nb_TP) + as.numeric(nzv_nb_FN)) * 
                (as.numeric(nzv_nb_TN) + as.numeric(nzv_nb_FP)) * 
                (as.numeric(nzv_nb_TN) + as.numeric(nzv_nb_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_nb_FP) * as.numeric(nzv_nb_FN))
part2_2 <- (as.numeric(nzv_nb_TP) * as.numeric(nzv_nb_TN))
part3_2 <- sqrt((as.numeric(nzv_nb_FP) + as.numeric(nzv_nb_TP)) * 
                  (as.numeric(nzv_nb_FP) + as.numeric(nzv_nb_TN)) * 
                  (as.numeric(nzv_nb_FN) + as.numeric(nzv_nb_TP)) * 
                  (as.numeric(nzv_nb_FN) + as.numeric(nzv_nb_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_nb_prob <- predict(nzv_nb_over, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_strat$Class, nzv_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NZV Naive Bayes Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9836
#Neg Pred Value/Precision Yes: 0.2095
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3378904
#MCC Yes: -0.3378904
#ROC: 0.8729

##______________________________________________________________________________##

##_____  Naive Bayes Model Both NZV Features _____##
set.seed(141)

nzv_nb_both <- naive_bayes(Class ~ ., data = both_train_nzv)
nzv_nb_both 
plot(nzv_nb_both)

nzv_nb_p_2 <- predict(nzv_nb_both, test_strat)
nzv_nb_cm_2 <- confusionMatrix(nzv_nb_p_2, test_strat$Class)
nzv_nb_cm_2

#TPR No Class
nzv_nb_TP_2 <- (nzv_nb_cm_2$table[1,1]) 
nzv_nb_FN_2 <- (nzv_nb_cm_2$table[2,1]) 
nzv_nb_TP_2
nzv_nb_FN_2
nzv_nb_TPR_2 <- nzv_nb_TP_2 / (nzv_nb_TP_2 + nzv_nb_FN_2)
print(nzv_nb_TPR_2)
#TPR: 0.73125

#TNR Yes Class
nzv_nb_TN_2 <- (nzv_nb_cm_2$table[2,2]) 
nzv_nb_FP_2 <- (nzv_nb_cm_2$table[1,2])
nzv_nb_TN_2
nzv_nb_FP_2
nzv_nb_TNR_2 <- (nzv_nb_TN_2 / (nzv_nb_TN_2 + nzv_nb_FP_2))
print(nzv_nb_TNR_2)
#TPR: 0.8412698

#FPR No Class
nzv_nb_FPR_2 <- nzv_nb_FP_2 / (nzv_nb_FP_2 + nzv_nb_TN_2)
print(nzv_nb_FPR_2)
#No FPR: 0.1587302

#FPR Yes Class
nzv_nb_FPR_2 <- nzv_nb_FN_2 / (nzv_nb_FN_2 + nzv_nb_TP_2)
print(nzv_nb_FPR_2)
#Yes FPR: 0.25

#Precisison No Class
precision_2 <- (nzv_nb_TP_2 / (nzv_nb_TP_2 + nzv_nb_FP_2))
#Pos Pred Value/Precision No: 0.9836 
recall_2 <- (nzv_nb_TN_2 / (nzv_nb_TN_2 + nzv_nb_FP_2))

F1 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_nb_FP_2 / (nzv_nb_FP_2 + nzv_nb_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.1978
recall_2 <- (nzv_nb_FN_2 / (nzv_nb_FN_2 + nzv_nb_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_nb_TP_2) * as.numeric(nzv_nb_TN_2))
part2 <- (as.numeric(nzv_nb_FP_2) * as.numeric(nzv_nb_FN_2))
part3 <- sqrt((as.numeric(nzv_nb_TP_2) + as.numeric(nzv_nb_FP_2)) * 
                (as.numeric(nzv_nb_TP_2) + as.numeric(nzv_nb_FN_2)) * 
                (as.numeric(nzv_nb_TN_2) + as.numeric(nzv_nb_FP_2)) * 
                (as.numeric(nzv_nb_TN_2) + as.numeric(nzv_nb_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_nb_FP_2) * as.numeric(nzv_nb_FN_2))
part2_2 <- (as.numeric(nzv_nb_TP_2) * as.numeric(nzv_nb_TN_2))
part3_2 <- sqrt((as.numeric(nzv_nb_FP_2) + as.numeric(nzv_nb_TP_2)) * 
                  (as.numeric(nzv_nb_FP_2) + as.numeric(nzv_nb_TN_2)) * 
                  (as.numeric(nzv_nb_FN_2) + as.numeric(nzv_nb_TP_2)) * 
                  (as.numeric(nzv_nb_FN_2) + as.numeric(nzv_nb_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_nb_prob <- predict(nzv_nb_both, test_strat, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc_both <- roc(test_strat$Class, nzv_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc_both, main = "ROC Curve for NZV Naive Bayes Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc_both <- auc(roc_both)
print(roc_auc_both)
#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Neg Pred Value/Precision Yes: 0.1978
#Pos Pred Value/Precision No: 0.9836
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3218696
#MCC Yes: -0.3218696
#ROC: 0.8646

##______________________________________________________________________________##
##_____  NZV Weighted RF Model Over df _____##
## Match Test & Train Features ##
nzv_over_features <- colnames(over_train_nzv)
nzv_over_features
nzv_test <- test_strat %>% select(all_of(nzv_over_features))
#dim(nzv_test)
#dim(over_train_nzv)

set.seed(223)
rf_over_train_nzv<- randomForest(Class ~., data = over_train_nzv, sampsize = c("Yes" = 185, "No" = 140), importance = TRUE)
rf_over_train_nzv
nzv_p_over <- predict(rf_over_train_nzv, nzv_test)
nzv_wrf_cm <- confusionMatrix(nzv_p_over, nzv_test$Class)
nzv_wrf_cm

#TPR No Class
nzv_wrf_TP <- (nzv_wrf_cm$table[1,1]) 
nzv_wrf_FN <- (nzv_wrf_cm$table[2,1]) 
nzv_wrf_TP
nzv_wrf_FN
nzv_wrf_TPR <- nzv_wrf_TP / (nzv_wrf_TP + nzv_wrf_FN)
print(nzv_wrf_TPR)
#TPR: 0.80375

#TNR Yes Class
nzv_wrf_TN <- (nzv_wrf_cm$table[2,2]) 
nzv_wrf_FP <- (nzv_wrf_cm$table[1,2])
nzv_wrf_TN
nzv_wrf_FP
nzv_wrf_TNR <- (nzv_wrf_TN / (nzv_wrf_TN + nzv_wrf_FP))
print(nzv_wrf_TNR)
#TPR: 0.8253968

#FPR No Class
nzv_wrf_FPR <- nzv_wrf_FP / (nzv_wrf_FP + nzv_wrf_TN)
print(nzv_wrf_FPR)
#No FPR: 0.1904762

#FPR Yes Class
nzv_wrf_FPR_2 <- nzv_wrf_FN / (nzv_wrf_FN + nzv_wrf_TP)
print(nzv_wrf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (nzv_wrf_TP / (nzv_wrf_TP + nzv_wrf_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (nzv_wrf_TN / (nzv_wrf_TN + nzv_wrf_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_wrf_FP / (nzv_wrf_FP + nzv_wrf_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (nzv_wrf_FN / (nzv_wrf_FN + nzv_wrf_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_wrf_TP) * as.numeric(nzv_wrf_TN))
part2 <- (as.numeric(nzv_wrf_FP) * as.numeric(nzv_wrf_FN))
part3 <- sqrt((as.numeric(nzv_wrf_TP) + as.numeric(nzv_wrf_FP)) * 
                (as.numeric(nzv_wrf_TP) + as.numeric(nzv_wrf_FN)) * 
                (as.numeric(nzv_wrf_TN) + as.numeric(nzv_wrf_FP)) * 
                (as.numeric(nzv_wrf_TN) + as.numeric(nzv_wrf_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_wrf_FP) * as.numeric(nzv_wrf_FN))
part2_2 <- (as.numeric(nzv_wrf_TP) * as.numeric(nzv_wrf_TN))
part3_2 <- sqrt((as.numeric(nzv_wrf_FP) + as.numeric(nzv_wrf_TP)) * 
                  (as.numeric(nzv_wrf_FP) + as.numeric(nzv_wrf_TN)) * 
                  (as.numeric(nzv_wrf_FN) + as.numeric(nzv_wrf_TP)) * 
                  (as.numeric(nzv_wrf_FN) + as.numeric(nzv_wrf_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_wrf_prob <- predict(rf_over_train_nzv, nzv_test, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nzv_test$Class, nzv_wrf_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NZV Weigthed Random Forest Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.8150
#Yes TPR/Recall: 0.8095
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9819
#Neg Pred Value/Precision Yes: 0.2563
#F1 No: 0.8874299
#F1 Yes: 0.03292791
#Kappa : 0.2475
#MCC No: 0.3857036
#MCC Yes: -0.3857036
#ROC: 0.8785

##______________________________________________________________________________##

##_____  NZV Weighted RF Model Both df _____##
## Match Test & Train Features ##
nzv_both_features <- colnames(both_train_nzv)
nzv_both_features
nzv_test <- test_strat %>% select(all_of(nzv_both_features))
#dim(nzv_test)
#dim(both_train_nzv)

set.seed(223)
rf_both_train_nzv<- randomForest(Class ~., data = both_train_nzv, sampsize = c("Yes" = 185, "No" = 140), importance = TRUE)
rf_both_train_nzv
nzv_p_both <- predict(rf_both_train_nzv, nzv_test)
nzv_wrf_cm <- confusionMatrix(nzv_p_both, nzv_test$Class)
nzv_wrf_cm

#TPR No Class
nzv_wrf_TP <- (nzv_wrf_cm$table[1,1]) 
nzv_wrf_FN <- (nzv_wrf_cm$table[2,1]) 
nzv_wrf_TP
nzv_wrf_FN
nzv_wrf_TPR <- nzv_wrf_TP / (nzv_wrf_TP + nzv_wrf_FN)
print(nzv_wrf_TPR)
#TPR: 0.80375

#TNR Yes Class
nzv_wrf_TN <- (nzv_wrf_cm$table[2,2]) 
nzv_wrf_FP <- (nzv_wrf_cm$table[1,2])
nzv_wrf_TN
nzv_wrf_FP
nzv_wrf_TNR <- (nzv_wrf_TN / (nzv_wrf_TN + nzv_wrf_FP))
print(nzv_wrf_TNR)
#TPR: 0.8253968

#FPR No Class
nzv_wrf_FPR <- nzv_wrf_FP / (nzv_wrf_FP + nzv_wrf_TN)
print(nzv_wrf_FPR)
#No FPR: 0.1904762

#FPR Yes Class
nzv_wrf_FPR_2 <- nzv_wrf_FN / (nzv_wrf_FN + nzv_wrf_TP)
print(nzv_wrf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (nzv_wrf_TP / (nzv_wrf_TP + nzv_wrf_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (nzv_wrf_TN / (nzv_wrf_TN + nzv_wrf_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_wrf_FP / (nzv_wrf_FP + nzv_wrf_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (nzv_wrf_FN / (nzv_wrf_FN + nzv_wrf_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_wrf_TP) * as.numeric(nzv_wrf_TN))
part2 <- (as.numeric(nzv_wrf_FP) * as.numeric(nzv_wrf_FN))
part3 <- sqrt((as.numeric(nzv_wrf_TP) + as.numeric(nzv_wrf_FP)) * 
                (as.numeric(nzv_wrf_TP) + as.numeric(nzv_wrf_FN)) * 
                (as.numeric(nzv_wrf_TN) + as.numeric(nzv_wrf_FP)) * 
                (as.numeric(nzv_wrf_TN) + as.numeric(nzv_wrf_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_wrf_FP) * as.numeric(nzv_wrf_FN))
part2_2 <- (as.numeric(nzv_wrf_TP) * as.numeric(nzv_wrf_TN))
part3_2 <- sqrt((as.numeric(nzv_wrf_FP) + as.numeric(nzv_wrf_TP)) * 
                  (as.numeric(nzv_wrf_FP) + as.numeric(nzv_wrf_TN)) * 
                  (as.numeric(nzv_wrf_FN) + as.numeric(nzv_wrf_TP)) * 
                  (as.numeric(nzv_wrf_FN) + as.numeric(nzv_wrf_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_wrf_prob <- predict(rf_both_train_nzv, nzv_test, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nzv_test$Class, nzv_wrf_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NZV Weigthed Random Forest Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.8150
#Yes TPR/Recall: 0.8095
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9819
#Neg Pred Value/Precision Yes: 0.2563
#F1 No: 0.8874299
#F1 Yes: 0.03292791
#Kappa : 0.2475
#MCC No: 0.3857036
#MCC Yes: -0.3857036
#ROC: 0.8702

##______________________________________________________________________________##

##_____  NN Model On NZV Over Features _____##
set.seed(223)

over_train_nzv$Class <- as.factor(over_train_nzv$Class)

# min-max normalization to [0, 1]
over_nzv_scaled <- over_train_nzv %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))
head(over_nzv_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
nzv_over_features <- colnames(over_train_nzv)
nzv_over_features
nn_test_nzv <- test_strat %>% select(all_of(nzv_over_features))
nn_test_nzv_scaled <- over_train_nzv %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(over_nzv_scaled)
dim(nn_test_nzv_scaled)

# build model with three units in hidden layer
nnet_over_model <- nnet(Class ~ ., data = over_nzv_scaled, size = 5, trace = TRUE)
nnet_over_model
plotnet(nnet_over_model)

# test the model
pred <- predict(nnet_over_model, newdata = nn_test_nzv_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_nzv_scaled$Class))

levels(pred)
levels(nn_test_nzv_scaled$Class)
nzv_nn_cm <- confusionMatrix(pred, nn_test_nzv_scaled$Class)
nzv_nn_cm
#Accuracy : 0.8767  
#Sensitivity : 0.9397          
#Specificity : 0.8129 
#Kappa : 0.7532

nzv_nn_TP_2 <- (nzv_nn_cm$table[1,1]) 
nzv_nn_FN_2 <- (nzv_nn_cm$table[2,1]) 
nzv_nn_TP_2
nzv_nn_FN_2
nzv_nn_TPR_2 <- nzv_nn_TP_2 / (nzv_nn_TP_2 + nzv_nn_FN_2)
print(nzv_nn_TPR_2)
#TPR: 0.795

#TNR Yes Class
nzv_nn_TN_2 <- (nzv_nn_cm$table[2,2]) 
nzv_nn_FP_2 <- (nzv_nn_cm$table[1,2])
nzv_nn_TN_2
nzv_nn_FP_2
nzv_nn_TNR_2 <- (nzv_nn_TN_2 / (nzv_nn_TN_2 + nzv_nn_FP_2))
print(nzv_nn_TNR_2)
#TPR: 0.8095238

#FPR No Class
nzv_nn_FPR_2 <- nzv_nn_FP_2 / (nzv_nn_FP_2 + nzv_nn_TN_2)
print(nzv_nn_FPR_2)
#No FPR: 0.1904762

#FPR Yes Class
nzv_nn_FPR_2 <- nzv_nn_FN_2 / (nzv_nn_FN_2 + nzv_nn_TP_2)
print(nzv_nn_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (nzv_nn_TP_2 / (nzv_nn_TP_2 + nzv_nn_FP_2))
#Pos Pred Value/Precision No: 0.8874299
recall <- (nzv_nn_TN_2 / (nzv_nn_TN_2 + nzv_nn_FP_2))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_nn_FP_2 / (nzv_nn_FP_2 + nzv_nn_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (nzv_nn_FN_2 / (nzv_nn_FN_2 + nzv_nn_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_nn_TP_2) * as.numeric(nzv_nn_TN_2))
part2 <- (as.numeric(nzv_nn_FP_2) * as.numeric(nzv_nn_FN_2))
part3 <- sqrt((as.numeric(nzv_nn_TP_2) + as.numeric(nzv_nn_FP_2)) * 
                (as.numeric(nzv_nn_TP_2) + as.numeric(nzv_nn_FN_2)) * 
                (as.numeric(nzv_nn_TN_2) + as.numeric(nzv_nn_FP_2)) * 
                (as.numeric(nzv_nn_TN_2) + as.numeric(nzv_nn_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_nn_FP_2) * as.numeric(nzv_nn_FN_2))
part2_2 <- (as.numeric(nzv_nn_TP_2) * as.numeric(nzv_nn_TN_2))
part3_2 <- sqrt((as.numeric(nzv_nn_FP_2) + as.numeric(nzv_nn_TP_2)) * 
                  (as.numeric(nzv_nn_FP_2) + as.numeric(nzv_nn_TN_2)) * 
                  (as.numeric(nzv_nn_FN_2) + as.numeric(nzv_nn_TP_2)) * 
                  (as.numeric(nzv_nn_FN_2) + as.numeric(nzv_nn_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_nn_prob <- predict(nnet_over_model, nn_test_nzv_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_nzv_scaled$Class, nzv_nn_prob)  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NZV Neural Net Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.9515

##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

##_____ Parameter tuning _____##
#ctrl <- trainControl(method = "CV", number = 10,
#                     summaryFunction = twoClassSummary,
#                     classProbs = TRUE,
#                     savePredictions = TRUE)
#
#nnetGrid <- expand.grid(size = 1:13, decay = seq(0, 2, 0.2))

#set.seed(223)
#nnetFit <- train(x = over_nzv_scaled[, -36], 
#                 y = over_nzv_scaled$Class,
#                 method = "nnet",
#                 metric = "ROC",
#                 preProc = c("center", "scale"),
#                 tuneGrid = nnetGrid,
#                 trace = FALSE,
#                 maxit = 100,
#                MaxNWts = 1000,
#                 trControl = ctrl)
#nnetFit
#nnetFit$bestTune
#plot(nnetFit)

#plotnet(nnetFit$finalModel)

#test_pred <- predict(nnetFit, newdata = nn_test_nzv_scaled)
#test_pred

#performance_measures <- confusionMatrix(test_pred, nn_test_nzv_scaled$Class)
#performance_measures
#Accuracy : 0.9756 
#Sensitivity : 0.9575        
#Specificity : 0.9940
#Kappa : 0.9513

##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##

##______________________________________________________________________________##

##_____  NN Model On NZV Both Features _____##
set.seed(223)

both_train_nzv$Class <- as.factor(both_train_nzv$Class)

# min-max normalization to [0, 1]
both_nzv_scaled <- both_train_nzv %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))
head(both_nzv_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
nzv_both_features <- colnames(both_train_nzv)
nzv_both_features
nn_test_nzv <- test_strat %>% select(all_of(nzv_both_features))
nn_test_nzv_scaled <- both_train_nzv %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(both_nzv_scaled)
dim(nn_test_nzv_scaled)

# build model with three units in hidden layer
nnet_both_model <- nnet(Class ~ ., data = both_nzv_scaled, size = 5, trace = TRUE)
nnet_both_model
plotnet(nnet_both_model)

# test the model
pred <- predict(nnet_both_model, newdata = nn_test_nzv_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_nzv_scaled$Class))

levels(pred)
levels(nn_test_nzv_scaled$Class)
nzv_nn_cm_2 <- confusionMatrix(pred, nn_test_nzv_scaled$Class)
nzv_nn_cm_2
#Accuracy : 0.8596 
#Sensitivity : 0.8661          
#Specificity : 0.8531 

nzv_nn_TP_2 <- (nzv_nn_cm_2$table[1,1]) 
nzv_nn_FN_2 <- (nzv_nn_cm_2$table[2,1]) 
nzv_nn_TP_2
nzv_nn_FN_2
nzv_nn_TPR_2 <- nzv_nn_TP_2 / (nzv_nn_TP_2 + nzv_nn_FN_2)
print(nzv_nn_TPR_2)
#TPR: 0.795

#TNR Yes Class
nzv_nn_TN_2 <- (nzv_nn_cm_2$table[2,2]) 
nzv_nn_FP_2 <- (nzv_nn_cm_2$table[1,2])
nzv_nn_TN_2
nzv_nn_FP_2
nzv_nn_TNR_2 <- (nzv_nn_TN_2 / (nzv_nn_TN_2 + nzv_nn_FP_2))
print(nzv_nn_TNR_2)
#TPR: 0.8095238

#FPR No Class
nzv_nn_FPR_2 <- nzv_nn_FP_2 / (nzv_nn_FP_2 + nzv_nn_TN_2)
print(nzv_nn_FPR_2)
#No FPR: 0.1904762

#FPR Yes Class
nzv_nn_FPR_2 <- nzv_nn_FN_2 / (nzv_nn_FN_2 + nzv_nn_TP_2)
print(nzv_nn_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (nzv_nn_TP_2 / (nzv_nn_TP_2 + nzv_nn_FP_2))
#Pos Pred Value/Precision No: 0.8874299
recall <- (nzv_nn_TN_2 / (nzv_nn_TN_2 + nzv_nn_FP_2))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (nzv_nn_FP_2 / (nzv_nn_FP_2 + nzv_nn_TP_2))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (nzv_nn_FN_2 / (nzv_nn_FN_2 + nzv_nn_TP_2))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(nzv_nn_TP_2) * as.numeric(nzv_nn_TN_2))
part2 <- (as.numeric(nzv_nn_FP_2) * as.numeric(nzv_nn_FN_2))
part3 <- sqrt((as.numeric(nzv_nn_TP_2) + as.numeric(nzv_nn_FP_2)) * 
                (as.numeric(nzv_nn_TP_2) + as.numeric(nzv_nn_FN_2)) * 
                (as.numeric(nzv_nn_TN_2) + as.numeric(nzv_nn_FP_2)) * 
                (as.numeric(nzv_nn_TN_2) + as.numeric(nzv_nn_FN_2)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(nzv_nn_FP_2) * as.numeric(nzv_nn_FN_2))
part2_2 <- (as.numeric(nzv_nn_TP_2) * as.numeric(nzv_nn_TN_2))
part3_2 <- sqrt((as.numeric(nzv_nn_FP_2) + as.numeric(nzv_nn_TP_2)) * 
                  (as.numeric(nzv_nn_FP_2) + as.numeric(nzv_nn_TN_2)) * 
                  (as.numeric(nzv_nn_FN_2) + as.numeric(nzv_nn_TP_2)) * 
                  (as.numeric(nzv_nn_FN_2) + as.numeric(nzv_nn_TN_2)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
nzv_nn_prob <- predict(nnet_both_model, nn_test_nzv_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_nzv_scaled$Class, nzv_nn_prob)  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NZV Neural Net Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.9734

##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
## Parameter tuning ##
set.seed(223)
#ctrl <- trainControl(method = "CV", number = 10,
#                     summaryFunction = twoClassSummary,
#                     classProbs = TRUE,
#                     savePredictions = TRUE)

#nnetGrid <- expand.grid(size = 1:13, decay = seq(0, 2, 0.2))

#nnetFit <- train(x = both_nzv_scaled[, -36], 
#                 y = both_nzv_scaled$Class,
#                 method = "nnet",
#                 metric = "ROC",
#                 preProc = c("center", "scale"),
#                 tuneGrid = nnetGrid,
#                 trace = FALSE,
#                 maxit = 100,
#                 MaxNWts = 1000,
#                 trControl = ctrl)
#nnetFit
#nnetFit$bestTune
#plot(nnetFit)

#plotnet(nnetFit$finalModel)

#test_pred <- predict(nnetFit, newdata = nn_test_nzv_scaled)
#test_pred

#performance_measures <- confusionMatrix(test_pred, nn_test_nzv_scaled$Class)
#performance_measures
## Best Performance ##
#Accuracy : 0.9645
#Sensitivity : 0.9419        
#Specificity : 0.9873
#Kappa : 0.9289
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
##______________________________ERROR____HERE!!!!!!!!____________________________________________##
#logisticc----------------------------------------
result <- logistic_regression_evaluation(over_nzv, test_over_nzv, "Class")
print(result)
result1 <- logistic_regression_evaluation(both_nzv, test_both_nzv, "Class")
print(result1)

#decision tree---------------------------
resdt <- decision_tree_evaluation(over_nzv, test_over_nzv, "Class")
print(resdt)
resdt1 <- decision_tree_evaluation(both_nzv, test_both_nzv, "Class")
print(resdt1)

#svm----------------------------------------------
resultsvmn <- svm_evaluation(over_nzv, test_over_nzv, "Class")
print(resultsvmn)
resultsvmn1 <- svm_evaluation(both_nzv, test_both_nzv, "Class")
print(resultsvmn1)


##______________________________________________________________________________##
##______________________________________________________________________________##

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____ Correlation with target variable _____##
## Function to compute and sort correlation with CLASS variable ##
correlation_with_class <- function(data, target_var) {
  # Ensure the target variable exists
  if (!(target_var %in% colnames(data))) {
    stop("Target variable not found in the dataset.")
  }
  
  # Select only numeric features
  numeric_data <- data %>% select(where(is.numeric))
  
  # Ensure target variable is numeric for correlation calculation
  if (!is.numeric(data[[target_var]])) {
    stop("Target variable must be numeric for correlation calculation.")
  }
  
  # Compute correlation with the target variable
  correlations <- cor(numeric_data, data[[target_var]], use = "complete.obs")
  
  # Convert to a dataframe for readability
  correlation_df <- data.frame(Feature = rownames(correlations), Correlation = correlations[, 1])
  
  # Remove the target variable itself from the results
  correlation_df <- correlation_df[correlation_df$Feature != target_var, ]
  
  # Sort by absolute correlation values in descending order
  correlation_df <- correlation_df %>% arrange(desc(abs(Correlation)))
  
  return(correlation_df)
}

## Compute and sort correlation for Over ##
over_train$Class <- as.numeric(as.factor(over_train$Class))
over_train_corr <- correlation_with_class(over_train, "Class")
print("Correlation with CLASS in Over dataset:")
print(over_train_corr)

## Compute and sort correlation for Both ##
both_train$Class <- as.numeric(as.factor(both_train$Class))
both_train_corr <- correlation_with_class(both_train, "Class")
print("Correlation with CLASS in Both dataset:")
print(both_train_corr)
dim(over_train_corr)
dim(both_train_corr)


##______________________________________________________________________________##
##_____  Naive Bayes Model Over Correlation Features _____##
set.seed(140)

corr_over_features <- over_train_corr$Feature
corr_over_features
over_train_corr <- over_train %>% select(all_of(corr_over_features))
test_corr <- test_strat %>% select(all_of(corr_over_features))

over_train_corr$Class <- as.factor(over_train$Class)
test_corr$Class <- as.factor(test_strat$Class)
dim(over_train_corr)
dim(test_corr)

corr_nb_over <- naive_bayes(Class ~ ., data = test_corr)
corr_nb_over 
#plot(corr_nb_over)

corr_nb_p <- predict(corr_nb_over, test_corr)
corr_nb_cm <- confusionMatrix(corr_nb_p, test_corr$Class)
corr_nb_cm

#TPR No Class
corr_nb_TP <- (corr_nb_cm$table[1,1]) 
corr_nb_FN <- (corr_nb_cm$table[2,1]) 
corr_nb_TP
corr_nb_FN
corr_nb_TPR <- corr_nb_TP / (corr_nb_TP + corr_nb_FN)
print(corr_nb_TPR)
#TPR: 0.75

#TNR Yes Class
corr_nb_TN <- (corr_nb_cm$table[2,2]) 
corr_nb_FP <- (corr_nb_cm$table[1,2])
corr_nb_TN
corr_nb_FP
corr_nb_TNR <- (corr_nb_TN / (corr_nb_TN + corr_nb_FP))
print(corr_nb_TNR)
#TPR: 0.8412698

#FPR No Class
corr_nb_FPR <- corr_nb_FP / (corr_nb_FP + corr_nb_TN)
print(corr_nb_FPR)
#No FPR: 0.1587302

#FPR Yes Class
corr_nb_FPR <- corr_nb_FN / (corr_nb_FN + corr_nb_TP)
print(corr_nb_FPR)
#Yes FPR: 0.25

#Precisison No Class
precision <- (corr_nb_TP / (corr_nb_TP + corr_nb_FP))
#Pos Pred Value/Precision No: 0.9836 
recall <- (corr_nb_TN / (corr_nb_TN + corr_nb_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision <- (corr_nb_FP / (corr_nb_FP + corr_nb_TP))
precision
#Neg Pred Value/Precision Yes: 0.1978
recall <- (corr_nb_FN / (corr_nb_FN + corr_nb_TP))

F1 <- (2 * precision * recall) / (precision + recall)
F1

#MCC No
part1 <- (as.numeric(corr_nb_TP) * as.numeric(corr_nb_TN))
part2 <- (as.numeric(corr_nb_FP) * as.numeric(corr_nb_FN))
part3 <- sqrt((as.numeric(corr_nb_TP) + as.numeric(corr_nb_FP)) * 
                (as.numeric(corr_nb_TP) + as.numeric(corr_nb_FN)) * 
                (as.numeric(corr_nb_TN) + as.numeric(corr_nb_FP)) * 
                (as.numeric(corr_nb_TN) + as.numeric(corr_nb_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_nb_FP) * as.numeric(corr_nb_FN))
part2_2 <- (as.numeric(corr_nb_TP) * as.numeric(corr_nb_TN))
part3_2 <- sqrt((as.numeric(corr_nb_FP) + as.numeric(corr_nb_TP)) * 
                  (as.numeric(corr_nb_FP) + as.numeric(corr_nb_TN)) * 
                  (as.numeric(corr_nb_FN) + as.numeric(corr_nb_TP)) * 
                  (as.numeric(corr_nb_FN) + as.numeric(corr_nb_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_nb_prob <- predict(corr_nb_over, test_corr, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc_over <- roc(test_corr$Class, corr_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc_over, main = "ROC Curve for Corr Naive Bayes Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc_over <- auc(roc_over)
print(roc_auc_over)
#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Neg Pred Value/Precision Yes: 0.1978
#Pos Pred Value/Precision No: 0.9836
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3218696
#MCC Yes: -0.3218696
#ROC: 0.8828

##______________________________________________________________________________##
#logisticccc------------------------
# Load required libraries
library(caret)
library(pROC)
library(e1071)

# Define the function to perform logistic regression and calculate evaluation metrics
logistic_regression_evaluation1 <- function(train_data, test_data, target_class) {
  # Ensure the target variable is factor
  train_data[[target_class]] <- as.factor(train_data[[target_class]])
  test_data[[target_class]] <- as.factor(test_data[[target_class]])
  
  # Define the logistic regression model formula
  model_formula <- as.formula(paste(target_class, "~ ."))
  
  # Train the logistic regression model
  logistic_model <- glm(model_formula, data = train_data, family = "binomial")
  
  # Predict probabilities on the test set
  pred_prob <- predict(logistic_model, newdata = test_data, type = "response")
  
  # Convert probabilities to class predictions (threshold = 0.5)
  pred_class <- ifelse(pred_prob > 0.5, "Yes", "No")
  
  # Convert predictions to factor with correct levels
  pred_class <- factor(pred_class, levels = levels(test_data[[target_class]]))
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(pred_class, test_data[[target_class]])
  
  # Extract performance metrics
  precision <- conf_matrix$byClass["Pos Pred Value"]
  recall <- conf_matrix$byClass["Sensitivity"]
  f_measure <- 2 * ((precision * recall) / (precision + recall))
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  kappa_value <- conf_matrix$overall["Kappa"]
  
  # Compute ROC and AUC
  roc_curve <- roc(test_data[[target_class]], pred_prob, levels = rev(levels(test_data[[target_class]])), direction = "<")
  auc_value <- auc(roc_curve)
  
  # Compute Matthews Correlation Coefficient (MCC)
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[2, 1])
  FN <- as.numeric(conf_matrix$table[1, 2])
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc_value <- ifelse(denominator == 0, 0, numerator / denominator)
  
  # Return a list with all the performance metrics
  return(list(
    Precision = precision,
    Recall = recall,
    F_Measure = f_measure,
    ROC_AUC = auc_value,
    MCC = mcc_value,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Kappa = kappa_value
  ))
}
result5 <- logistic_regression_evaluation1(over_train_corr, test_corr, "Class")
print(result5)
#decision tree----------------------------------------------
resultcor <- decision_tree_evaluation(over_train_corr, test_corr, "Class")
print(resultcor)

#svm-----overr---------------------------------


# Load required libraries
library(caret)
library(pROC)
library(e1071)

# Define the function to perform SVM and calculate evaluation metrics
svm_evaluation <- function(train_data, test_data, target_class) {
  
  # Ensure the target variable is a factor
  train_data[[target_class]] <- as.factor(train_data[[target_class]])
  test_data[[target_class]] <- as.factor(test_data[[target_class]])
  
  # Make sure the factor levels are valid R variable names
  levels(train_data[[target_class]]) <- make.names(levels(train_data[[target_class]]))
  levels(test_data[[target_class]]) <- make.names(levels(test_data[[target_class]]))
  
  # Ensure no missing values in the target variable
  if (any(is.na(train_data[[target_class]])) | any(is.na(test_data[[target_class]]))) {
    stop("Target variable contains missing values. Please handle them before proceeding.")
  }
  
  # Define the SVM model formula
  model_formula <- as.formula(paste(target_class, "~ ."))
  
  # Train the SVM model using caret with the method 'svmRadial' for SVM with radial basis function kernel
  svm_model <- train(model_formula, data = train_data, method = "svmRadial", 
                     trControl = trainControl(method = "cv", number = 5, classProbs = TRUE))
  
  # Predict probabilities on the test set
  pred_prob <- predict(svm_model, newdata = test_data, type = "prob")
  
  # Convert predicted probabilities to class predictions (threshold = 0.5)
  pred_class <- ifelse(pred_prob[,2] > 0.5, levels(test_data[[target_class]])[2], levels(test_data[[target_class]])[1])
  
  # Convert predictions to factor with correct levels
  pred_class <- factor(pred_class, levels = levels(test_data[[target_class]]))
  
  # Compute confusion matrix
  conf_matrix <- confusionMatrix(pred_class, test_data[[target_class]])
  
  # Extract performance metrics
  precision   <- conf_matrix$byClass["Pos Pred Value"]
  recall      <- conf_matrix$byClass["Sensitivity"]
  f_measure   <- 2 * ((precision * recall) / (precision + recall))
  sensitivity <- conf_matrix$byClass["Sensitivity"]
  specificity <- conf_matrix$byClass["Specificity"]
  kappa_value <- conf_matrix$overall["Kappa"]
  
  # Compute ROC and AUC
  roc_curve <- roc(test_data[[target_class]], pred_prob[,2], levels = rev(levels(test_data[[target_class]])), direction = "<")
  auc_value <- auc(roc_curve)
  
  # Compute Matthews Correlation Coefficient (MCC)
  TP <- as.numeric(conf_matrix$table[2, 2])
  TN <- as.numeric(conf_matrix$table[1, 1])
  FP <- as.numeric(conf_matrix$table[1, 2])
  FN <- as.numeric(conf_matrix$table[2, 1])
  
  numerator <- (TP * TN) - (FP * FN)
  denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
  mcc_value <- ifelse(denominator == 0, 0, numerator / denominator)
  
  # Return a list with all the performance metrics
  return(list(
    Precision   = precision,
    Recall      = recall,
    F_Measure   = f_measure,
    ROC_AUC     = auc_value,
    MCC         = mcc_value,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Kappa       = kappa_value
  ))
}

# Example Usage:
# Assuming you have a dataset already split into train_data and test_data, and the target variable is named "target_variable".
result <- svm_evaluation(over_train_corr, test_corr, "Class")
print(result)

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____  Naive Bayes Model Both Correlation Features _____##
set.seed(141)

corr_both_features <- both_train_corr$Feature
corr_both_features
both_train_corr <- both_train %>% select(all_of(corr_both_features))
test_corr <- test_strat %>% select(all_of(corr_both_features))

both_train_corr$Class <- as.factor(both_train$Class)
test_corr$Class <- as.factor(test_strat$Class)
dim(both_train_corr)
dim(test_corr)

corr_nb_both <- naive_bayes(Class ~ ., data = test_corr)
corr_nb_both 
#plot(corr_nb_both)

corr_nb_p <- predict(corr_nb_both, test_corr)
corr_nb_cm <- confusionMatrix(corr_nb_p, test_corr$Class)
corr_nb_cm

#TPR No Class
corr_nb_TP <- (corr_nb_cm$table[1,1]) 
corr_nb_FN <- (corr_nb_cm$table[2,1]) 
corr_nb_TP
corr_nb_FN
corr_nb_TPR <- corr_nb_TP / (corr_nb_TP + corr_nb_FN)
print(corr_nb_TPR)
#TPR: 0.75

#TNR Yes Class
corr_nb_TN <- (corr_nb_cm$table[2,2]) 
corr_nb_FP <- (corr_nb_cm$table[1,2])
corr_nb_TN
corr_nb_FP
corr_nb_TNR <- (corr_nb_TN / (corr_nb_TN + corr_nb_FP))
print(corr_nb_TNR)
#TPR: 0.8412698

#FPR No Class
corr_nb_FPR <- corr_nb_FP / (corr_nb_FP + corr_nb_TN)
print(corr_nb_FPR)
#No FPR: 0.1587302

#FPR Yes Class
corr_nb_FPR <- corr_nb_FN / (corr_nb_FN + corr_nb_TP)
print(corr_nb_FPR)
#Yes FPR: 0.25

#Precisison No Class
precision <- (corr_nb_TP / (corr_nb_TP + corr_nb_FP))
#Pos Pred Value/Precision No: 0.9836 
recall <- (corr_nb_TN / (corr_nb_TN + corr_nb_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision <- (corr_nb_FP / (corr_nb_FP + corr_nb_TP))
precision
#Neg Pred Value/Precision Yes: 0.1978
recall <- (corr_nb_FN / (corr_nb_FN + corr_nb_TP))

F1 <- (2 * precision * recall) / (precision + recall)
F1

#MCC No
part1 <- (as.numeric(corr_nb_TP) * as.numeric(corr_nb_TN))
part2 <- (as.numeric(corr_nb_FP) * as.numeric(corr_nb_FN))
part3 <- sqrt((as.numeric(corr_nb_TP) + as.numeric(corr_nb_FP)) * 
                (as.numeric(corr_nb_TP) + as.numeric(corr_nb_FN)) * 
                (as.numeric(corr_nb_TN) + as.numeric(corr_nb_FP)) * 
                (as.numeric(corr_nb_TN) + as.numeric(corr_nb_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_nb_FP) * as.numeric(corr_nb_FN))
part2_2 <- (as.numeric(corr_nb_TP) * as.numeric(corr_nb_TN))
part3_2 <- sqrt((as.numeric(corr_nb_FP) + as.numeric(corr_nb_TP)) * 
                  (as.numeric(corr_nb_FP) + as.numeric(corr_nb_TN)) * 
                  (as.numeric(corr_nb_FN) + as.numeric(corr_nb_TP)) * 
                  (as.numeric(corr_nb_FN) + as.numeric(corr_nb_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_nb_prob <- predict(corr_nb_both, test_corr, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc_both <- roc(test_corr$Class, corr_nb_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc_both, main = "ROC Curve for corr Naive Bayes Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc_both <- auc(roc_both)
print(roc_auc_both)
#No TPR/Recall: 0.75
#Yes TPR/Recall: 0.8412698
#No FPR: 0.1587302
#Yes FPR: 0.25
#Neg Pred Value/Precision Yes: 0.1978
#Pos Pred Value/Precision No: 0.9836
#F1 No: 0.9068872
#F1 Yes: 0.03076923
#Kappa : 0.2475
#MCC No: 0.3218696
#MCC Yes: -0.3218696
#ROC: 0.8828

##______________________________________________________________________________##
##______________________________________________________________________________##


#logistic--------------
result6 <- logistic_regression_evaluation1(both_train_corr, test_corr, "Class")
print(result6)
#decison tree-----------------------
resultcor1 <- decision_tree_evaluation(both_train_corr, test_corr, "Class")
print(resultcor1)

#svmm-----bothh---------------------------------------
resultsvm11 <- svm_evaluation(both_train_corr, test_corr, "Class")
print(resultsvm11)

##______________________________________________________________________________##
##______________________________________________________________________________##

##_____  Correlation RF Model Over df _____##
## Match Test & Train Features ##
#set.seed(223)
#corr_over_features <- over_train_corr$Feature
#corr_over_features
#over_train_corr <- over_train %>% select(all_of(corr_over_features))
#test_corr <- test_strat %>% select(all_of(corr_over_features))

#over_train_corr$Class <- as.factor(over_train$Class)
#test_corr$Class <- as.factor(test_strat$Class)
#dim(over_train_corr)
#dim(test_corr)

set.seed(223)
rf_over_train_corr<- randomForest(Class ~., data = over_train_corr, importance = TRUE)
rf_over_train_corr
corr_p <- predict(rf_over_train_corr, test_corr)
levels(corr_p) <- c("No", "Yes")  # Adjust according to your levels
levels(test_corr$Class) <- c("No", "Yes")  # Adjust according to your levels
corr_cm <- confusionMatrix(corr_p, test_corr$Class)
corr_cm
levels(corr_p)
table(test_corr$Class)
#TPR No Class
corr_rf_TP <- (corr_cm$table[1,1]) 
corr_rf_FN <- (corr_cm$table[2,1]) 
corr_rf_TP
corr_rf_FN
corr_rf_TPR <- corr_rf_TP / (corr_rf_TP + corr_rf_FN)
print(corr_rf_TPR)
#TPR: 0.98375

#TNR Yes Class
corr_rf_TN <- (corr_cm$table[2,2]) 
corr_rf_FP <- (corr_cm$table[1,2])
corr_rf_TN
corr_rf_FP
corr_rf_TNR <- (corr_rf_TN / (corr_rf_TN + corr_rf_FP))
print(corr_rf_TNR)
#TPR: 0.4285714

#FPR No Class
corr_rf_FPR <- corr_rf_FP / (corr_rf_FP + corr_rf_TN)
print(corr_rf_FPR)
#No FPR: 0.1904762

#FPR Yes Class
corr_rf_FPR_2 <- corr_rf_FN / (corr_rf_FN + corr_rf_TP)
print(corr_rf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (corr_rf_TP / (corr_rf_TP + corr_rf_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (corr_rf_TN / (corr_rf_TN + corr_rf_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (corr_rf_FP / (corr_rf_FP + corr_rf_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (corr_rf_FN / (corr_rf_FN + corr_rf_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(corr_rf_TP) * as.numeric(corr_rf_TN))
part2 <- (as.numeric(corr_rf_FP) * as.numeric(corr_rf_FN))
part3 <- sqrt((as.numeric(corr_rf_TP) + as.numeric(corr_rf_FP)) * 
                (as.numeric(corr_rf_TP) + as.numeric(corr_rf_FN)) * 
                (as.numeric(corr_rf_TN) + as.numeric(corr_rf_FP)) * 
                (as.numeric(corr_rf_TN) + as.numeric(corr_rf_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_rf_FP) * as.numeric(corr_rf_FN))
part2_2 <- (as.numeric(corr_rf_TP) * as.numeric(corr_rf_TN))
part3_2 <- sqrt((as.numeric(corr_rf_FP) + as.numeric(corr_rf_TP)) * 
                  (as.numeric(corr_rf_FP) + as.numeric(corr_rf_TN)) * 
                  (as.numeric(corr_rf_FN) + as.numeric(corr_rf_TP)) * 
                  (as.numeric(corr_rf_FN) + as.numeric(corr_rf_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_rf_prob <- predict(rf_over_train_corr, test_corr, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_corr$Class, corr_rf_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for RF Weigthed Random Forest Over Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.8150
#Yes TPR/Recall: 0.8095
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9819
#Neg Pred Value/Precision Yes: 0.2563
#F1 No: 0.8874299
#F1 Yes: 0.03292791
#Kappa : 0.2475
#MCC No: 0.3857036
#MCC Yes: -0.3857036
#ROC: 0.8663


##______________________________________________________________________________##

##_____  Correlation RF Model Both df _____##
## Match Test & Train Features ##
#corr_both_features <- both_train_corr$Feature
#corr_both_features
#both_train_corr <- both_train %>% select(all_of(corr_both_features))
#test_corr <- test_strat %>% select(all_of(corr_both_features))

#both_train_corr$Class <- as.factor(both_train$Class)
#test_corr$Class <- as.factor(test_strat$Class)
#dim(both_train_corr)
#dim(test_corr)

set.seed(223)
rf_both_train_corr<- randomForest(Class ~., data = both_train_corr, importance = TRUE)
rf_both_train_corr
corr_p <- predict(rf_both_train_corr, test_corr)
levels(corr_p) <- c("No", "Yes")  # Adjust according to your levels
levels(test_corr$Class) <- c("No", "Yes")  # Adjust according to your levels
corr_cm <- confusionMatrix(corr_p, test_corr$Class)
corr_cm
levels(corr_p)
table(test_corr$Class)
#TPR No Class
corr_rf_TP <- (corr_cm$table[1,1]) 
corr_rf_FN <- (corr_cm$table[2,1]) 
corr_rf_TP
corr_rf_FN
corr_rf_TPR <- corr_rf_TP / (corr_rf_TP + corr_rf_FN)
print(corr_rf_TPR)
#TPR: 0.98375

#TNR Yes Class
corr_rf_TN <- (corr_cm$table[2,2]) 
corr_rf_FP <- (corr_cm$table[1,2])
corr_rf_TN
corr_rf_FP
corr_rf_TNR <- (corr_rf_TN / (corr_rf_TN + corr_rf_FP))
print(corr_rf_TNR)
#TPR: 0.4285714

#FPR No Class
corr_rf_FPR <- corr_rf_FP / (corr_rf_FP + corr_rf_TN)
print(corr_rf_FPR)
#No FPR: 0.1904762

#FPR Yes Class
corr_rf_FPR_2 <- corr_rf_FN / (corr_rf_FN + corr_rf_TP)
print(corr_rf_FPR_2)
#Yes FPR: 0.185

#Precisison No Class
precision <- (corr_rf_TP / (corr_rf_TP + corr_rf_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (corr_rf_TN / (corr_rf_TN + corr_rf_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision_2 <- (corr_rf_FP / (corr_rf_FP + corr_rf_TP))
precision_2
#Neg Pred Value/Precision Yes: 0.2095
recall_2 <- (corr_rf_FN / (corr_rf_FN + corr_rf_TP))

F1_2 <- (2 * precision_2 * recall_2) / (precision_2 + recall_2)
F1_2

#MCC No
part1 <- (as.numeric(corr_rf_TP) * as.numeric(corr_rf_TN))
part2 <- (as.numeric(corr_rf_FP) * as.numeric(corr_rf_FN))
part3 <- sqrt((as.numeric(corr_rf_TP) + as.numeric(corr_rf_FP)) * 
                (as.numeric(corr_rf_TP) + as.numeric(corr_rf_FN)) * 
                (as.numeric(corr_rf_TN) + as.numeric(corr_rf_FP)) * 
                (as.numeric(corr_rf_TN) + as.numeric(corr_rf_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_rf_FP) * as.numeric(corr_rf_FN))
part2_2 <- (as.numeric(corr_rf_TP) * as.numeric(corr_rf_TN))
part3_2 <- sqrt((as.numeric(corr_rf_FP) + as.numeric(corr_rf_TP)) * 
                  (as.numeric(corr_rf_FP) + as.numeric(corr_rf_TN)) * 
                  (as.numeric(corr_rf_FN) + as.numeric(corr_rf_TP)) * 
                  (as.numeric(corr_rf_FN) + as.numeric(corr_rf_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_rf_prob <- predict(rf_both_train_corr, test_corr, type = "prob")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(test_corr$Class, corr_rf_prob[, 1])  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for RF Weigthed Random Forest Both Model", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.8150
#Yes TPR/Recall: 0.8095
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9819
#Neg Pred Value/Precision Yes: 0.2563
#F1 No: 0.8874299
#F1 Yes: 0.03292791
#Kappa : 0.2475
#MCC No: 0.3857036
#MCC Yes: -0.3857036
#ROC: 0.8626

##______________________________________________________________________________##

##_____  NN Model On Correlation Over Features _____##
set.seed(223)
dim(over_train_corr)
dim(test_corr)

# min-max normalization to [0, 1]
over_corr_scaled <- over_train_corr %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))
head(over_corr_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
corr_over_features <- colnames(over_train_corr)
corr_over_features
nn_test_corr <- test_strat %>% select(all_of(corr_over_features))
nn_test_corr_scaled <- over_train_corr %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(over_corr_scaled)
dim(nn_test_corr_scaled)

# build model with three units in hidden layer
nnet_over_model <- nnet(Class ~ ., data = over_corr_scaled, size = 5, trace = TRUE)
nnet_over_model
plotnet(nnet_over_model)

# test the model
pred <- predict(nnet_over_model, newdata = nn_test_corr_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_corr_scaled$Class))

levels(pred)
levels(nn_test_corr_scaled$Class)
corr_nn_cm <- confusionMatrix(pred, nn_test_corr_scaled$Class)
corr_nn_cm
#Accuracy : 0.7734  
#Sensitivity : 0.6726          
#Specificity : 0.8756

corr_nn_TP <- (corr_nn_cm$table[1,1]) 
corr_nn_FN <- (corr_nn_cm$table[2,1]) 
corr_nn_TP
corr_nn_FN
corr_nn_TPR <- corr_nn_TP / (corr_nn_TP + corr_nn_FN)
print(corr_nn_TPR)
#TPR: 0.795

#TNR Yes Class
corr_nn_TN <- (corr_nn_cm$table[2,2]) 
corr_nn_FP <- (corr_nn_cm$table[1,2])
corr_nn_TN
corr_nn_FP
corr_nn_TNR <- (corr_nn_TN / (corr_nn_TN + corr_nn_FP))
print(corr_nn_TNR)
#TPR: 0.8095238

#FPR No Class
corr_nn_FPR <- corr_nn_FP / (corr_nn_FP + corr_nn_TN)
print(corr_nn_FPR)
#No FPR: 0.1904762

#FPR Yes Class
corr_nn_FPR <- corr_nn_FN / (corr_nn_FN + corr_nn_TP)
print(corr_nn_FPR)
#Yes FPR: 0.185

#Precisison No Class
precision <- (corr_nn_TP / (corr_nn_TP + corr_nn_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (corr_nn_TN / (corr_nn_TN + corr_nn_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision <- (corr_nn_FP / (corr_nn_FP + corr_nn_TP))
precision
#Neg Pred Value/Precision Yes: 0.2095
recall <- (corr_nn_FN / (corr_nn_FN + corr_nn_TP))

F1 <- (2 * precision * recall) / (precision + recall)
F1

#MCC No
part1 <- (as.numeric(corr_nn_TP) * as.numeric(corr_nn_TN))
part2 <- (as.numeric(corr_nn_FP) * as.numeric(corr_nn_FN))
part3 <- sqrt((as.numeric(corr_nn_TP) + as.numeric(corr_nn_FP)) * 
                (as.numeric(corr_nn_TP) + as.numeric(corr_nn_FN)) * 
                (as.numeric(corr_nn_TN) + as.numeric(corr_nn_FP)) * 
                (as.numeric(corr_nn_TN) + as.numeric(corr_nn_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_nn_FP) * as.numeric(corr_nn_FN))
part2_2 <- (as.numeric(corr_nn_TP) * as.numeric(corr_nn_TN))
part3_2 <- sqrt((as.numeric(corr_nn_FP) + as.numeric(corr_nn_TP)) * 
                  (as.numeric(corr_nn_FP) + as.numeric(corr_nn_TN)) * 
                  (as.numeric(corr_nn_FN) + as.numeric(corr_nn_TP)) * 
                  (as.numeric(corr_nn_FN) + as.numeric(corr_nn_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_nn_prob <- predict(nnet_over_model, nn_test_corr_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_corr_scaled$Class, corr_nn_prob)  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NN Correlation Over Features", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.8272


##______________________________________________________________________________##

##_____  NN Model On Correlation Both Features _____##
set.seed(223)

# min-max normalization to [0, 1]
both_corr_scaled <- over_train_corr %>% mutate(across(ESR:RACPI, ~ ( . - min(.)) / (max(.) - min(.))))
head(both_corr_scaled)

## Match Test & Train Features ##
# Ensure that 'Class' in the test set is a factor
corr_both_features <- colnames(both_train_corr)
corr_both_features
nn_test_corr <- test_strat %>% select(all_of(corr_both_features))
nn_test_corr_scaled <- both_train_corr %>% mutate(across(PUMA:WAOB, ~ ( . - min(.)) / (max(.) - min(.))))

dim(both_corr_scaled)
dim(nn_test_corr_scaled)

# build model with three units in hidden layer
nnet_both_model <- nnet(Class ~ ., data = both_corr_scaled, size = 5, trace = TRUE)
nnet_both_model
plotnet(nnet_both_model)

# test the model
pred <- predict(nnet_both_model, newdata = nn_test_corr_scaled, type = "class")
pred <- factor(pred, levels = levels(nn_test_corr_scaled$Class))

levels(pred)
levels(nn_test_corr_scaled$Class)
corr_nn_cm <- confusionMatrix(pred, nn_test_corr_scaled$Class)
corr_nn_cm
#Accuracy : 0.7734  
#Sensitivity : 0.6726          
#Specificity : 0.8756

corr_nn_TP <- (corr_nn_cm$table[1,1]) 
corr_nn_FN <- (corr_nn_cm$table[2,1]) 
corr_nn_TP
corr_nn_FN
corr_nn_TPR <- corr_nn_TP / (corr_nn_TP + corr_nn_FN)
print(corr_nn_TPR)
#TPR: 0.795

#TNR Yes Class
corr_nn_TN <- (corr_nn_cm$table[2,2]) 
corr_nn_FP <- (corr_nn_cm$table[1,2])
corr_nn_TN
corr_nn_FP
corr_nn_TNR <- (corr_nn_TN / (corr_nn_TN + corr_nn_FP))
print(corr_nn_TNR)
#TPR: 0.8095238

#FPR No Class
corr_nn_FPR <- corr_nn_FP / (corr_nn_FP + corr_nn_TN)
print(corr_nn_FPR)
#No FPR: 0.1904762

#FPR Yes Class
corr_nn_FPR <- corr_nn_FN / (corr_nn_FN + corr_nn_TP)
print(corr_nn_FPR)
#Yes FPR: 0.185

#Precisison No Class
precision <- (corr_nn_TP / (corr_nn_TP + corr_nn_FP))
#Pos Pred Value/Precision No: 0.8874299
recall <- (corr_nn_TN / (corr_nn_TN + corr_nn_FP))

F1 <- (2 * precision * recall) / (precision + recall)
F1
#F1 No: 0.9068872

#Precisison Yes Class
precision <- (corr_nn_FP / (corr_nn_FP + corr_nn_TP))
precision
#Neg Pred Value/Precision Yes: 0.2095
recall <- (corr_nn_FN / (corr_nn_FN + corr_nn_TP))

F1 <- (2 * precision * recall) / (precision + recall)
F1

#MCC No
part1 <- (as.numeric(corr_nn_TP) * as.numeric(corr_nn_TN))
part2 <- (as.numeric(corr_nn_FP) * as.numeric(corr_nn_FN))
part3 <- sqrt((as.numeric(corr_nn_TP) + as.numeric(corr_nn_FP)) * 
                (as.numeric(corr_nn_TP) + as.numeric(corr_nn_FN)) * 
                (as.numeric(corr_nn_TN) + as.numeric(corr_nn_FP)) * 
                (as.numeric(corr_nn_TN) + as.numeric(corr_nn_FN)))

mcc <- (part1 - part2) / part3
mcc

#MCC Yes
part1_2 <- (as.numeric(corr_nn_FP) * as.numeric(corr_nn_FN))
part2_2 <- (as.numeric(corr_nn_TP) * as.numeric(corr_nn_TN))
part3_2 <- sqrt((as.numeric(corr_nn_FP) + as.numeric(corr_nn_TP)) * 
                  (as.numeric(corr_nn_FP) + as.numeric(corr_nn_TN)) * 
                  (as.numeric(corr_nn_FN) + as.numeric(corr_nn_TP)) * 
                  (as.numeric(corr_nn_FN) + as.numeric(corr_nn_TN)))

mcc_2 <- (part1_2 - part2_2) / part3_2
mcc_2

#ROC No
# Predict class probabilities using the trained Naive Bayes model
corr_nn_prob <- predict(nnet_both_model, nn_test_corr_scaled, type = "raw")

# Assuming the "No" class corresponds to the first column in the output (adjust if necessary)
roc <- roc(nn_test_corr_scaled$Class, corr_nn_prob)  # Probabilities for the "No" class

# Plot the ROC curve for the "No" class
plot(roc, main = "ROC Curve for NN Correlation Both Features", col = "blue", lwd = 2)

# Calculate and print the AUC (Area Under Curve)
roc_auc <- auc(roc)
print(roc_auc)

#No TPR/Recall: 0.7937
#Yes TPR/Recall: 0.8254
#No FPR: 0.1587302
#Yes FPR: 0.25
#Pos Pred Value/Precision No: 0.9830
#Neg Pred Value/Precision Yes: 0.2396
#F1 No: 0.8973192
#F1 Yes: 0.03145853
#Kappa : 0.2912 
#MCC No: 0.3712469
#MCC Yes: -0.3712469
#ROC: 0.7872

##______________________________________________________________________________##
##______________________________________________________________________________##
##______________________________________________________________________________##
##____________________________________DONE!!____________________________________##










