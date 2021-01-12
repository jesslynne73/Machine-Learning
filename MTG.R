# Author: Jess Strait

# Front matter
rm(list=ls())
library(data.table)
library(Metrics)
library(caret)
library(glmnet)

# Intake data
train <- fread('final_train.csv')
test <- fread('final_test.csv')
set <- fread('set_tab.csv')
cards <- fread('card_tab.csv')

# Create master table for feature engineering
test$future_price <- 0
# Generate separator column
test$train <- 0
train$train <- 1
master <- rbind(train, test)

# Set keys for future merges
setkey(master, id)
setkey(cards, id)

# Initial basic feature engineering example from class
cards$Legendary <- 0
cards$Legendary[grep('Legendary', cards$supertypes)] <- 1

# Feature engineering for rarity categories
rarity <- as.data.table(tstrsplit(cards$rarity, ' '))
rarity$id <- cards$id
m_rarity <- melt(rarity, id.vars = 'id')
m_rarity <- data.table(m_rarity)
m_rarity <- m_rarity[!is.na(m_rarity$value)]
m_rarity$True <- 1
m_rarity <- dcast(m_rarity, id ~ value, length, value.var='True')
setkey(rarity, id)

# Merge new features to master table
master <- merge(master, cards[,.(id,Legendary)], all.x= TRUE)
master <- merge(master, m_rarity, all.x= TRUE)

# Clean up master table as shown in class
master$current_price[is.na(master$current_price)] <- mean(master$current_price, na.rm = TRUE)

# Feature engineering for cmc categories
cmc <- as.data.table(tstrsplit(cards$cmc, ' '))
cmc$id <- cards$id
m_cmc <- melt(cmc, id.vars = 'id')
m_cmc <- data.table(m_cmc)
m_cmc <- m_cmc[!is.na(m_cmc$value)]
m_cmc$True <- 1
m_cmc <- dcast(m_cmc, id ~ value, length, value.var='True')
setkey(cmc, id)
master <- merge(master, m_cmc, all.x= TRUE)

# Feature engineering for colors categories
colors <- as.data.table(tstrsplit(cards$colors, ' '))
colors$id <- cards$id
m_colors <- melt(colors, id.vars = 'id')
m_colors <- data.table(m_colors)
m_colors <- m_colors[!is.na(m_colors$value)]
m_colors$True <- 1
m_colors <- dcast(m_colors, id ~ value, length, value.var='True')
setkey(colors, id)
master <- merge(master, m_colors, all.x= TRUE)
master$Black[is.na(master$Black)] <- 0
master$Blue[is.na(master$Blue)] <- 0
master$Green[is.na(master$Green)] <- 0
master$Red[is.na(master$Red)] <- 0
master$White[is.na(master$White)] <- 0

# Split data back into train and test
train <- master[train == 1]
test <- master[train == 0]

# Remove dummy columns
train$train <- NULL
test$train <- NULL
test$future_price <- NULL

# Write new data to interim folder
fwrite(train, "train_interim.csv")
fwrite(test, "test_interim.csv")

# Restore variables for dummies
train_y <- train$future_price
test$future_price <- 0
test_y <- test$future_price

# Save ID variables before removal
trainid <- train$id
testid <- test$id
train$id <- NULL
test$id <- NULL

# Generate dummies and store as matrices for modeling
dummies <- dummyVars(future_price ~ ., data = train)
traindummies <- predict(dummies, newdata = train)
testdummies <- predict(dummies, newdata = test)
x <- as.matrix(traindummies)
x_test <- as.matrix(testdummies)

# Generate starter lambdas
lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression with cross validation through glmnet
lasso_reg <- cv.glmnet(x, train_y, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

# Identify best lambda
lambda_best <- lasso_reg$lambda.min 
lambda_best

# Adapt model to use best lambda
lasso_model <- glmnet(x, train_y, alpha = 1, lambda = lambda_best, standardize = TRUE)

# Generate and evaluate training predictions
predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
rmse(train_y, predictions_train)

# Generate testing predictions
predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)

# Save model
summary(lasso_model)
saveRDS(lasso_model, "master_lasso.model")

# Create submission file
test$id <- testid
test$future_price <- predictions_test
submit <- test[,.(id, future_price)]
fwrite(submit,"submit_mtg2_final.csv")
