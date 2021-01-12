# Author: Jess Strait

# Front Matter
rm(list = ls())
library(data.table)
library(Metrics)

# Import and explore data
train <- fread("Stat_380_train.csv")
test <- fread("Stat_380_test.csv")
View(train)
class(train)

# Fill NA values with mean of column 
train$LotFrontage <- as.numeric(train$LotFrontage)
test$LotFrontage <- as.numeric(test$LotFrontage)
train[, LotFrontage := replace(LotFrontage, is.na(LotFrontage), 0)]
test[, LotFrontage := replace(LotFrontage, is.na(LotFrontage), 0)]

# Feature engineering: age of house
train$age <- as.numeric(train$YrSold - train$YearBuilt)
test$age <- as.numeric(test$YrSold - test$YearBuilt)

# Write new data to interim folder
fwrite(train,"trainengineer.csv")
fwrite(test, "testengineer.csv")

# Front Matter
rm(list = ls())
library(data.table)
library(Metrics)
library(caret)
library(glmnet)
library(xgboost)

# Import and explore data
train <- fread("trainengineer.csv")
test <- fread("testengineer.csv")

# Initialize variables for dummies
train_y <- train$SalePrice
test$SalePrice <- 0
test_y <- test$SalePrice

# Save ID variables before removal
trainid <- train$Id
testid <- test$Id
train$Id <- NULL
test$Id <- NULL

# Generate dummies and store as matrices for modeling
dummies <- dummyVars(SalePrice ~ ., data = train)
traindummies <- predict(dummies, newdata = train)
testdummies <- predict(dummies, newdata = test)
x <- as.matrix(traindummies)
x_test <- as.matrix(testdummies)

# Explore alpha and lambda combinations for penalized regression

# Setting alpha = 1 implements lasso regression with cross validation
lasso_reg <- cv.glmnet(x, train_y, alpha = 1, family="gaussian")

# Identify best lambda
lambda_best <- lasso_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
lasso_model <- glmnet(x, train_y, alpha = 1, lambda = lambda_best, family="gaussian")

# Generate and evaluate training predictions
predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
rmse(train_y, predictions_train)
# 25218.7

# Try other elastic net approaches with alpha = 0.6

# Setting alpha = 0.6
elastic_reg <- cv.glmnet(x, train_y, alpha = 0.6, family="gaussian")

# Identify best lambda
lambda_best <- elastic_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
elastic_model <- glmnet(x, train_y, alpha = 0.6, lambda = lambda_best, family="gaussian")

# Generate and evaluate training predictions
predictions_train <- predict(elastic_model, s = lambda_best, newx = x)
rmse(train_y, predictions_train)
# 25220.32

# Try with alpha = 0.8

# Setting alpha = 0.8
elastic_reg <- cv.glmnet(x, train_y, alpha = 0.8, family="gaussian")

# Identify best lambda
lambda_best <- elastic_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
elastic_model <- glmnet(x, train_y, alpha = 0.8, lambda = lambda_best, family="gaussian")

# Generate and evaluate training predictions
predictions_train <- predict(elastic_model, s = lambda_best, newx = x)
rmse(train_y, predictions_train)
# 25218.9

# Try with alpha = 0.2

# Setting alpha = 0.2
elastic_reg <- cv.glmnet(x, train_y, alpha = 0.2, family="gaussian")

# Identify best lambda
lambda_best <- elastic_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
elastic_model <- glmnet(x, train_y, alpha = 0.2, lambda = lambda_best, family="gaussian")

# Generate and evaluate training predictions
predictions_train <- predict(elastic_model, s = lambda_best, newx = x)
rmse(train_y, predictions_train)
# 25218.8

# Try ridge regression with alpha = 0

# Setting alpha = 0
ridge_reg <- cv.glmnet(x, train_y, alpha = 0, family="gaussian")

# Identify best lambda
lambda_best <- ridge_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
ridge_model <- glmnet(x, train_y, alpha = 0, lambda = lambda_best, family="gaussian")

# Generate and evaluate training predictions
predictions_train <- predict(ridge_model, s = lambda_best, newx = x, type="response")
rmse(train_y, predictions_train)
# 2244.33

# Alpha = 1 gave the lowest RMSE and we will proceed with that model.

# Setting alpha = 1
final_reg <- cv.glmnet(x, train_y, alpha = 1, family="gaussian")

# Identify best lambda
lambda_best <- final_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
final_model <- glmnet(x, train_y, alpha = 1, lambda = lambda_best, family="gaussian")

# Generate testing predictions
predictions_test <- predict(final_model, s = lambda_best, newx = x_test, type="response")

# Save model
summary(final_model)
saveRDS(final_model, "master_final.model")

# Create submission file
test$Id <- testid
test$SalePrice <- predictions_test
submit <- test[,.(Id, SalePrice)]
fwrite(submit,"submission_final.csv")

# Now explore XGBoost
boosttrain <- xgb.DMatrix(traindummies,label=train_y,missing=NA)
boosttest <- xgb.DMatrix(testdummies,missing=NA)

# Use cross validation to identify tuning parameters as shown in class 
tuning <- NULL

# Combinations of parameters tested
# gamma = 0.002, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15630.36
# gamma = 0.002, eta = .002, max_depth = 10, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15917.03
# gamma = 0.002, eta = .002, max_depth = 15, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15568.09
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15429.37
# gamma = 0.002, eta = .1, max_depth = 15, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15698.66
# gamma = 0.002, eta = .05, max_depth = 15, min_child_weight = 1, subsample = 1, colsample_bytree = 1, test-rmse = 15588.01
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 5, subsample = 1, colsample_bytree = 1, test-rmse = 15518.65
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = 1, colsample_bytree = 1, test-rmse = 15352.95
# gamma = 0.01, eta = .01, max_depth = 15, min_child_weight = 2, subsample = 1, colsample_bytree = 1, test-rmse = 15478.37
# gamma = 0.0001, eta = .01, max_depth = 15, min_child_weight = 2, subsample = 1, colsample_bytree = 1, test-rmse = 15359.76
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = 1, test-rmse = 15179.88
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .9, colsample_bytree = 1, test-rmse = 15258.05
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .9, test-rmse = 15124.55
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .8, test-rmse = 15116.60
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .7, test-rmse = 15083.91
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .5, test-rmse = 15422.06
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .6, test-rmse = 15326.38

# The best parameters we've found are
# gamma = 0.002, eta = .01, max_depth = 15, min_child_weight = 2, subsample = .95, colsample_bytree = .7, test-rmse = 15083.91

parameters <- list(  objective           = "reg:squarederror",
                     gamma               = 0.002,
                     booster             = "gbtree",
                     eval_metric         = "rmse",
                     eta                 = 0.01,
                     max_depth           = 15,
                     min_child_weight    = 2,
                     subsample           = .95,
                     colsample_bytree    = .7,
                     tree_method = 'hist'
)

XGBm <- xgb.cv(params=parameters,nfold=5,nrounds=10000,missing=NA,data=boosttrain,print_every_n=1, early_stopping_rounds=25)

# Save best results
iter_results <- data.table(t(parameters), best_iter = XGBm$best_iteration, rmse = XGBm$evaluation_log$test_rmse_mean[XGBm$best_iteration])
tuning <- rbind(tuning, iter_results)
fwrite(tuning, "besttuning.csv")

# Fit the model to training data
watchlist <- list(train = boosttrain)
XGBm <- xgb.train(params=parameters,nrounds=143,missing=NA,data=boosttrain,watchlist=watchlist,print_every_n=1)

# Generate and evaluate testing predictions
pred <- predict(XGBm, newdata = boosttest)
rmse(test_y, pred)

# XGBoost did not perform as well as the regression model. We will submit the results from that model to Kaggle.

# Save XGBoost model
summary(XGBm)
saveRDS(XGBm, "boost.model")

# Create submission file
testlast <- NULL
testlast$Id <- testid
test$SalePrice <- pred
submitlast <- test[,.(Id, SalePrice)]
fwrite(submitlast,"submission_XGB.csv")

