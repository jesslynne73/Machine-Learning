# Base-Lift Split Analysis Full Stack
# Author: Jess Strait for CROPP Cooperative

# Clears environment of any pre-existing values
rm(list = ls())

# REQUIRED USER INPUTS -------------------------------------------------------------
# Note: All CSV files must be uploaded to your Home working directory.

# Reports: Your dataset(s) from ISM with Base & Lift information. Add additional lines if more reports will be needed.
report1 <- read.csv("Report1.csv")
report2 <- read.csv("Report2.csv", header=FALSE)
report3 <- read.csv("Report3.csv", header=FALSE)
report4 <- read.csv("Report4.csv", header=FALSE)

# Modify this code chunk to the correct number of CSV files you will be inputting. You may comment out code with the # symbol or remove a line if it is not needed. 
colnames(report2) <- colnames(report1)
colnames(report3) <- colnames(report1)
colnames(report4) <- colnames(report1)
data <- rbind(report1, report2, report3, report4)

# Jgatlas: Your dataset from JG Atlas with demand's Base & Lift prediction.
jgatlas <- read.csv("JG_Atlas_Promo_Lift_2017-2021.csv")
# Test_start: The first day of the period you are predicting in YY-MM-DD format. For example, if you wish to predict 2020, start_date <- "20-01-01"
test_start <- "20-01-01"

# Test_end: The last day of the period you are predicting in YY-MM-DD format. For example, if you wish to predict 2020, end_date <- "20-12-31"
test_end <- "20-12-31"

# Name: The name of the CSV file you wish to write your results to ending in .csv. For example, "results.csv"
file_name <- "BaseLiftSplitAnalysisDampedTRUE.csv"


# FRONT MATTER (MAY REQUIRE USE INPUT FOR INSTALLATION) ----------------------------------------------------------------
print("Loading front matter...")
# Sets working directory to home folder
setwd("~/")
# Loads packages: use command install.packages("packagename") if you do not have one installed already
# All these packages offer many functions and tools, just giving an idea here
library(dplyr) # Used for basic data frame manipulation
library(forecast) # Used for exponential smoothing and other forecasting methods
library(lubridate) # Used for manipulating date-time objects
library(stringr) # Used for identifying patterns in character vectors
library(aTSA) # Used for time series analysis and forecasting
library(xgboost) # Used for extreme gradient boosting
library(caret) # Used to predict with a constructed model
library(stats) # Used for many statistical functions
library(Metrics) # Used for model evaluation
library(MLmetrics) # Also used for model evaluation


# DATA CLEANING -----------------------------------------------------------

print("Cleaning data for join operation...")

# Select relevant variables from JG Atlas data
jgatlas <- jgatlas %>% select(Product.Key, Week.Begin.Date, Promotion.Lift.Quantity.UOM, Promotion.Count)

# Clean JG Atlas date = to match ISM date for joining
jgatlas$year <- year(as.Date(jgatlas$Week.Begin.Date, format="%m/%d/%Y"))
# Removes first two digits of year (i.e. "2017" becomes "17")
str_sub(jgatlas$year, 1, 2) <- ''
jgatlas$month <- month(as.Date(jgatlas$Week.Begin.Date, format="%m/%d/%Y"))
jgatlas$day <- day(as.Date(jgatlas$Week.Begin.Date, format="%m/%d/%Y"))

# Reconstruct JG Atlas date variable to match formatting of ISM date
i <- 1
for (i in 1:nrow(jgatlas)){
  if (nchar(jgatlas$month[[i]]) < 2){
    jgatlas$month[[i]] <- paste("0", jgatlas$month[[i]], sep='')
  }
  if (nchar(jgatlas$day[[i]]) < 2){
    jgatlas$day[[i]] <- paste("0", jgatlas$day[[i]], sep='')
  }
  jgatlas$Sales.Date[[i]] <- paste(jgatlas$month[[i]], "/", jgatlas$day[[i]], "/", jgatlas$year[[i]], sep='')
  i <- i+1
}

# Remove incorrectly formatted date and cast new date to character data type
jgatlas$Week.Begin.Date <- NULL
jgatlas$Sales.Date <- as.character(jgatlas$Sales.Date)

# Create joined dataframe of ISM and JG Atlas data by joining on product key and sales date
joined <- inner_join(data, jgatlas, by=c("Sales.Date", "Product.Key"))
joined <- rename(joined, CMDTY = Commodity.Key, FMC = Forecast.Management.Code, Date = Sales.Date, BaseVolumeUOM = Base.Volume.UOM, BItoShipDiff = B.I.to.Ship.Diff, PromotionalLiftUOM = Promotional.Lift.UOM, OrderQtyUOM = Order.Qty.UOM, ShipQtyUOM = Ship.Qty.UOM, BItoShipCheck = B.I.to.Ship.Check, BItoShipPercentDiff = B.I.to.Ship...Diff, TradeDollars = Trade.., EDLP = EDLP.., PromoCount = Low.Level.Promo.Count..No.EDLP., DemandPromoLift = Promotion.Lift.Quantity.UOM, DemandPromoCount = Promotion.Count)

# REMOVE HASHTAG from this line if you are UNSURE which columns need formatting to be numeric type.
# lapply(joined, class)

# Format all columns to be coerced as numeric
joined$BaseVolumeUOM <- as.numeric(gsub(",","",joined$BaseVolumeUOM))
joined$PromotionalLiftUOM <- as.numeric(gsub(",","",joined$PromotionalLiftUOM))
joined$OrderQtyUOM <- as.numeric(gsub(",","",joined$OrderQtyUOM))
joined$ShipQtyUOM <- as.numeric(gsub(",","",joined$ShipQtyUOM))
joined$EDLP <- gsub("\\$","",joined$EDLP)
joined$EDLP <- as.numeric(gsub(",","",joined$EDLP))
joined$DemandPromoLift <- as.numeric(gsub(",","",joined$DemandPromoLift))

# Drop extracted date variables as we will redo this later when constructing seasonality
joined$year <- NULL
joined$month <- NULL
joined$day <- NULL

# Fill NA rows with 0 where demand reported no predicted lift or promo counts
joined$DemandPromoLift[is.na(joined$DemandPromoLift)] <- 0
joined$DemandPromoCount[is.na(joined$DemandPromoCount)] <- 0

# Construct variables representing the difference in ISM and Demand's predicted lift and promo counts
joined$PromoDiff <- joined$PromotionalLiftUOM - joined$DemandPromoLift
joined$PromoCountDiff <- joined$PromoCount - joined$DemandPromoCount

# Construct shorting variable where shipping quantity is less than order quantity
i <- 1
for (i in 1:nrow(joined)){
  if (joined$ShipQtyUOM[[i]] < joined$OrderQtyUOM[[i]]){
    joined$Shorting[[i]] <- 1
  }
  else{
    joined$Shorting[[i]] <- 0
  }
  i <- i+1
}
joined$Shorting <- as.numeric(joined$Shorting)

print("Dropping invalid products...")
# Instantiates empty list
x <- list()

# Rename dataframe, saving joined version as it is if needed
data <- joined

# Drops all invalid products
# Identifies where Product is Inactive, appends them to empty list
i <- 1
for (i in 1:nrow(data)){
  # str_detect searches for a particular string in the product name column, and adds it to the droplist if it's present
  if (str_detect(data$Product.Name[[i]], "INACT")) {
    x <- append(x, data$Product.Name[[i]])
    i <- i+1
    next
  }
  if (str_detect(data$FMC[[i]], "Do Not Plan")) {
    x <- append(x, data$FMC[[i]])
    i <- i+1
    next
  }
  if (str_detect(data$Product.Name[[i]], "Planning")) {
    x <- append(x, data$Product.Name[[i]])
    i <- i+1
    next
  }
}
# Creates droplist of unique products for removal
droplist <- as.list(unique(unlist(x)))

# Creates list of dataframes grouped by product name
groups <- data %>% group_by(Product.Name) %>% group_split()

# Removes products in droplist or with no shipping quantity over time period
i <- 1
for (i in 1:length(groups)){
  # Remove groups with product name present in droplist
  if (groups[[i]]$Product.Name %in% droplist){
    groups[[i]] <- NULL
  }
  # Remove groups with zero shipped units over entire time period
  if (sum(groups[[i]]$ShipQtyUOM) == 0){
    groups[[i]] <- NULL
  }
  i <- i+1
}

# Constructs dataframe of remaining products
data <- bind_rows(groups)

# Creates list of dataframes grouped by FMC
groups <- data %>% group_by(FMC) %>% group_split()

# Removes FMCs in droplist or with no shipping quantity over time period
i <- 1
for (i in 1:length(groups)){
  # Remove groups with FMC present in droplist
  if (groups[[i]]$FMC %in% droplist){
    groups[[i]] <- NULL
  }
}

# Constructs dataframe of remaining FMCs
data <- bind_rows(groups)

print("Generating dummy variables...")
# Extracts information from date variable
data$month <- month(as.POSIXlt(data$Date, format="%m/%d/%Y"))
data$year <- year(as.POSIXlt(data$Date, format="%m/%d/%Y"))

# Initializes seasonality dummies
# Cast to time series and try decomp()
data$summer <- 0
data$spring <- 0
data$fall <- 0
data$winter <- 0

# Updates seasonality dummies based on month
i <- 1
for (i in 1:nrow(data)){
  if (data$month[[i]] < 3 || data$month[[i]] == 12){
    data$winter[[i]] <- 1
    i <- i+1
    next
  }
  if ( 2 < data$month[[i]] & data$month[[i]] < 6){
    data$spring[[i]] <- 1
    i <- i+1
    next
  }
  if ( 8 < data$month[[i]] & data$month[[i]] < 12){
    data$fall[[i]] <- 1
    i <- i+1
    next
  }
  if ( 5 < data$month[[i]] & data$month[[i]] < 9){
    data$summer[[i]] <- 1
    i <- i+1
    next
  }
}  

# Clears date variable and prepares month/year variables for dummy conversion
date_vector <- as.data.frame(data$Date)
# Set key based on row number to join date variable back in later
date_vector$key <- seq.int(nrow(date_vector))
data$Date <- NULL
data$month <- as.factor(data$month)
data$year <- as.factor(data$year)

# Creates dummy variables from dataframe
dummies <- dummyVars(~ ., data = data)
numdummies <- as.data.frame(predict(dummies, newdata = data))
# Set key
data$key <- seq.int(nrow(data))
# Set key
numdummies$key <- seq.int(nrow(numdummies))


# XGBOOST MODELING --------------------------------------------------------

print("Beginning modeling...")
# Prepares data for modeling with train-test-split
# Test: whatever year(s) you are predicting - finds dummies column of that year and filters where true
test <- numdummies %>% filter(numdummies[ , grepl(year(test_start), names(numdummies))] == 1)
# Train: whatever year(s) you are showing the model during training - finds dumies column of that year and filters where false
train <- numdummies %>% filter(numdummies[ , grepl(year(test_start), names(numdummies))] == 0)
# x is independent variables, y is dependent variable (base volume)
# XGBoost requires matrix data structure for training
y_train <- train %>% select(BaseVolumeUOM)
y_train <- as.matrix(y_train)
y_test <- test %>% select(BaseVolumeUOM)
y_test <- as.matrix(y_test)
x_train <- subset(train, select = -c(BaseVolumeUOM, key))
x_train <- as.matrix(x_train)
x_test <- subset(test, select = -c(BaseVolumeUOM, key))
x_test <- as.matrix(x_test)

# Fits XGBoost model
# Additional parameters are available and can all be tuned: https://xgboost.readthedocs.io/en/latest/parameter.html
# data: independent variables (x)
# label: dependent variable (y)
# verbosity: what intermittent steps are printed to the console (set to 0 for silent, 1 for eval at every iteration, or 3 for all info)
# max_depth: maximum depth of decision tree (number of levels)
# objective: count:poisson ensures only positive predictions # Don't change
# colsample_bytree: ratio of columns selected for fitting each tree
# eta: learning rate
# alpha: weight regularization term
# min_child_weight: minimum number of occurrences at each tree leaf
# nrounds: number of trees fitted and new weights applied
# eval_metric: can be changed but mae = mean absolute error
xgb <- xgboost(data=x_train, label=y_train, verbosity=1, max_depth=4, objective="count:poisson", colsample_bytree = 1, eta=.05, alpha=5, min_child_weight=3, nrounds=11000, eval.metric='mae')

# Makes predictions on model unseen testing data
test_preds <- predict(xgb, x_test)
# Computes mean absolute error & R-Squared for model evaluation
mae(y_test, test_preds)
R2_Score(test_preds, y_test)

print("Saving model predictions...")
# Reconstructs dataframe with predictions
train$PredictedBase <- NaN
test$PredictedBase <- test_preds
data2 <- rbind(train, test)
# Joins back in the date vector
data2 <- inner_join(data2, date_vector, by="key")
# Selects out desired variables
data <- inner_join(data, data2, by="key")
data <- data %>% select(CMDTY, FMC, `data$Date`, PredictedBase, Product.Name, Product.Key.x, BaseVolumeUOM.x, EDLP.x, PromotionalLiftUOM.x, OrderQtyUOM.x, ShipQtyUOM.x, Shorting.x, BItoShipCheck.x, BItoShipDiff.x, BItoShipPercentDiff.x, TradeDollars.x, PromoCount.x, DemandPromoCount.x, DemandPromoLift.x, PromoDiff.x, PromoCountDiff.x)
# Rename variables
data <- rename(data, Date = `data$Date`, EDLP = EDLP.x, PromoCount = PromoCount.x, Product.Key = Product.Key.x, BaseVolumeUOM = BaseVolumeUOM.x, PromotionalLiftUOM = PromotionalLiftUOM.x, OrderQtyUOM = OrderQtyUOM.x, ShipQtyUOM = ShipQtyUOM.x, Shorting = Shorting.x, BItoShipCheck = BItoShipCheck.x, BItoShipDiff = BItoShipDiff.x, BItoShipPercentDiff = BItoShipPercentDiff.x, TradeDollars = TradeDollars.x, DemandPromoCount = DemandPromoCount.x, DemandPromoLift = DemandPromoLift.x, PromoDiff = PromoDiff.x, PromoCountDiff = PromoCountDiff.x)
# Create variable indicator to show when predicted base does not fit within business problem parameters
data$ErrorPred <- "NA"

# Show where predicted base is nonsensical: should not be below zero or greater than shipping quantity
# Remove pound signs from commented-out if statement lines to change predictions to be within reasonable range
for (i in 1:nrow(data)){
  if (is.na(data[i, ]$PredictedBase)){
    next
  }
  if (data[i,]$PredictedBase < 0){
    # data[i,]$PredictedBase <- 0
    data[i,]$ErrorPred <- "Negative"
  }
  if (data[i,]$PredictedBase > data[i,]$ShipQtyUOM){
    # data[i,]$PredictedBase <- data[i,]$ShipQtyUOM
    data[i,]$ErrorPred <- "Overpredict"
  }
  i <- i+1
}


# HOLT-WINTERS SMOOTHING --------------------------------------------------

print("Beginning Holt-Winters exponential smoothing...")
# Cleans up data
data$Date <- as.Date(data$Date, format="%m/%d/%Y")
# data$FMC <- str_replace(string=data$FMC, pattern="\\s+", replacement="")
# Save training data set for joining after smoothing
saved_train <- data %>% filter(Date > as.Date(test_end) | Date < as.Date(test_start))
# Constructs list of dataframes grouped by product name - only test data is relevant because it is the only data with predictions for smoothing
prod <- data %>% filter(Date >= as.Date(test_start) & Date <= as.Date(test_end)) %>% group_by(Product.Name) %>% group_split()

i <- 1
# Instantiates empty list to fill with dataframes
hw <- list()
# Damping factor for Holt()
# Converts each dataframe's predicted base to time series and conducts Holt-Winters exponential smoothing

for (i in 1:length(prod)){
  prod[[i]] <- prod[[i]] %>% arrange(Date)
  # Initialize start of time series to first date in dataframe
  start <- head(prod[[i]], 1)$Date
  # Convert predicted column to time series with weekly frequency (frequency = 52)
  ts <- ts(prod[[i]]$PredictedBase, frequency = 52, start = start)
  print(prod[[i]]$Product.Name)
  print(Holt(ts))
  print(Holt(ts)$estimate)
  hw[[i]] <- data.frame("HoltWinters" = Holt(ts, damped=TRUE)$estimate, "Date" = prod[[i]]$Date, "Product.Name" = prod[[i]]$Product.Name)
  i <- i+1
}  

# Recombines groups split by FMC and Holt Winters results split by FMC, then joins both frames
data_hw <- inner_join(bind_rows(prod), bind_rows(hw), by=c("Date", "Product.Name"))
saved_train$HoltWinters <- NaN
data3 <- rbind(saved_train, data_hw)

# Limit Holt Winters to reasonable restrictions: may not be below zero or greater than shipping quantity
for (i in 1:nrow(data3)){
  if (is.na(data3[i, ]$PredictedBase)){
    next
  }
  if (data3[i,]$HoltWinters < 0){
    data3[i,]$HoltWinters <- 0
  }
  if (data3[i,]$HoltWinters > data3[i,]$ShipQtyUOM){
    data3[i,]$HoltWinters <- data3[i,]$ShipQtyUOM
  }
  i <- i+1
}

# OFFSET 4 WK AVERAGE -----------------------------------------------------

# Computes offset 4 week average - previous two weeks, current week, and future week
print("Computing offset 4 week average...")

# Sorts dataframe by product name first, then by date ascending
sort <- data3 %>% arrange(Product.Name, as.Date(Date))
saved_train2 <- sort %>% filter(Date < as.Date(test_start) | Date > as.Date(test_end))
# Group split by product again
prod <- sort %>% filter(Date >= as.Date(test_start) & Date <= as.Date(test_end)) %>% group_by(Product.Name) %>% group_split()
# Instantiates empty list to fill with dataframes
offset <- list()

# Compute offset 4 week average
# Note: this code chunk SHOULD give error messages in the console. There is a try-error catch to handle this.
i <- 1
for (i in 1:length(prod)){
  prod_select <- prod[[i]] %>% arrange(Date)
  # Instantiate empty dataframe inside list
  offset[[i]] <- data.frame("Offset4WkAvg" = double(), "Date" = Date(), "Product.Name" = character())
  for (j in 1:nrow(prod_select)){
    # If calculating before row 3, there are not two previous weeks to calculate on
    if (j<3){
      # Calculate on current and next week to avoid NA
      values <- mean(prod_select$PredictedBase[[j]], prod_select$PredictedBase[[j+1]])
      # Add row to dataframe with offset average, date, and product
      offset[[i]][j,] <- list(values, prod_select[j,]$Date, prod_select[j,]$Product.Name)
      j <- j+1
      next
    }
    # Ideal case averages previous two weeks, current week, and next week
    values <- try(mean(prod_select$PredictedBase[[j]], prod_select$PredictedBase[[j-1]], prod_select$PredictedBase[[j+1]], prod_select$PredictedBase[[j-2]]))
    # Try-error occurs if trying to average with final week in FMC because there is no next week
    if (class(values)=='try-error'){
      # Calculate based on previous two weeks and current week
      values <- mean(prod_select$PredictedBase[[j]], prod_select$PredictedBase[[j-1]], prod_select$PredictedBase[[j-2]])
    }
    # Add row to dataframe with offset average, date, and product
    offset[[i]][j,] <- list(values, prod_select[j,]$Date, prod_select[j,]$Product.Name)
    j <- j+1
  }
  i <- i+1
}  

# Recombines groups split by product and offset four week average results split by product, then joins both frames
data_offset <- inner_join(bind_rows(prod), bind_rows(offset), by=c("Date", "Product.Name"))
saved_train2$Offset4WkAvg <- NaN
data_final <- rbind(saved_train2, data_offset)


# FINAL DATA CLEANING -----------------------------------------------------

print("Finishing up data cleaning...")

# Compute lift value by subtracting base from shipping quantity
data_final$PredictedLift <- data_final$ShipQtyUOM - data_final$PredictedBase
data_final <- data_final %>% arrange(Product.Name, Date)

# Compute average of three forecast terms
i <- 1
for (i in 1:nrow(data_final)){
  if(is.nan(data_final$PredictedBase[i])){
    data_final$`AvgofPredictedBase,HW,Offset4WkAvg`[i] <- "NA"
  } else {
    data_final$`AvgofPredictedBase,HW,Offset4WkAvg`[i] <- mean(data_final$PredictedBase[i], data_final$HoltWinters[i], data_final$Offset4WkAvg[i])
  }
  i <- i+1
}

# Convert all floats to integers (appropriate for unit volume)
data_final$PredictedBase <- as.integer(data_final$PredictedBase)
data_final$Offset4WkAvg <- as.integer(data_final$Offset4WkAvg)
data_final$HoltWinters <- as.integer(data_final$HoltWinters)
data_final$`AvgofPredictedBase,HW,Offset4WkAvg` <- as.integer(data_final$`AvgofPredictedBase,HW,Offset4WkAvg`)
data_final$PredictedLift <- as.integer(data_final$PredictedLift)


# Write to CSV, which is downloadable from your working directory
write.csv(data_final, file_name)

print("Results written to CSV in working directory.")
