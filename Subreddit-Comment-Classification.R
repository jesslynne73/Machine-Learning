# Author: Jess Strait
# Feature engineering of subreddit training data

# Front Matter
rm(list=ls())
library(data.table)
library(Rtsne)
library(caret)
library(ggplot2)

# Intake training data
train <- fread("train_data.csv")

# Create single response variable from ten dummy variables
train$response <- 0
train$response[grep('1', train$subredditcars)] <- 0
train$response[grep('1', train$subredditCooking)] <- 1
train$response[grep('1', train$subredditMachineLearning)] <- 2
train$response[grep('1', train$subredditmagicTCG)] <- 3
train$response[grep('1', train$subredditpolitics)] <- 4
train$response[grep('1', train$subredditReal_Estate)] <- 5
train$response[grep('1', train$subredditscience)] <- 6
train$response[grep('1', train$subredditStockMarket)] <- 7
train$response[grep('1', train$subreddittravel)] <- 8
train$response[grep('1', train$subredditvideogames)] <- 9

# Create final table with only ID and response
final <- data.table()
final$id <- train$id
final$text <- train$text
final$response <- train$response

# Intake all data, including engineered training data
train <- final
test <- fread("test_data.csv")
train_emb <- fread("train_emb.csv")
test_emb <- fread("test_emb.csv")

# Join embeddings to unstructured text data
train_id <- train$id
test_id <- test$id
train_emb$id <- train_id
test_emb$id <- test_id
train <- merge(train, train_emb, by='id')
test <- merge(test, test_emb, by='id')

# Make master dataset for engineering
# Initialize conditioning variables for dummies
train_response <- train$response
test$response <- 0
test_response <- test$response
# 0 is train
train$tt <- 0
# 1 is test
test$tt <- 1
data <- rbind(train, test)

# PCA

# Remove ID and text variables
data_id <- data$id
data$id <- NULL
data_text <- data$text
data$text <- NULL

# Create dummies variables
dummies <- dummyVars(response ~ ., data = data)
datadummies <- predict(dummies, newdata = data)

# Run and explore principal component analysis
pca <- prcomp(datadummies)
screeplot(pca)
summary(pca)
biplot(pca)

# Save principal component coordinates
pca_dt <- data.table(pca$x)

# tSNE

# tSNE at perplexity = 50 
# With optimal XGBoost: test-mlogloss = .169181 # .20522
set.seed(3)
# tSNE at perplexity = 20, test-mlogloss = .18608
# tSNE at perplexity = 40, test-mlogloss = .179127
# tSNE at perplexity = 60, test-mlogloss = .245442
# tSNE at perplexity = 70, test-mlogloss = .17899
# tSNE at perplexity = 90, test-mlogloss = .1947
perplexityvalue <- 70
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)

# Obtain tSNE coordinates and observe clustering
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

# Add other variables of interest to tSNE frame
tsne_dt$id <- data_id
tsne_dt$tt <- data$tt
tsne_dt$response <- data$response
tsne_dt$text <- data_text

# Do feature engineering to identify words and phrases from text
tsne_dt$carnames <- as.numeric(grepl("ford|tesla|mercedes|toyota|hyundai|honda|subaru|sedan|coupe|mustang|renault|nissan|porsche|minivan|model x|model s| model 3| model y|audi|escalade|cadillac|chevrolet|horsepower|bmw|vw|volkswagen|volvo|lexus|kia|prius|camry|corolla|suv|sonata|accord|mph|ferrari|lamborghini|jeep|mazda|hatchback|car", tsne_dt$text, ignore.case=TRUE))
tsne_dt$carwords <- as.numeric(grepl("drive|mechanic|tires|steering", tsne_dt$text, ignore.case=TRUE))
tsne_dt$food <- as.numeric(grepl("mayo|sugar|baking|vegan|vegetarian|meatless|casserole|barbecue|thanksgiving|flavor|carrot|onion|garlic|salt|mise en place|chicken|food|plate|bowl|spoon|herb|spice|knives|protein|vegetable|fruit|chef", tsne_dt$text, ignore.case=TRUE))
tsne_dt$cook <- as.numeric(grepl("cook|bake|baking|dinner|breakfast|lunch|meal|ingredient|recipe|oven|stove|knife", tsne_dt$text, ignore.case=TRUE))
tsne_dt$mlwords <- as.numeric(grepl("regression|gaussian|sklearn|predict|statistics|machine learning|ai|overfit|artificial intelligence|python|support vector machine|deep learning|data mining|neural net|SVM|pytorch|tensorflow|algorithm|linear|logistic|github|regressor|classifier|supervised", tsne_dt$text, ignore.case=TRUE))
tsne_dt$magicwords <- as.numeric(grepl("mana|cmc|mtg|card|deck|morph|magic|secret lair|reckoner|boros|dimir|simic|playmat|arcane|wotc|planeswalker", tsne_dt$text, ignore.case=TRUE))
tsne_dt$political <- as.numeric(grepl("politic|immigration|scotus|potus|census|healthcare|medicaid|medicare|obama|abortion|republican|democrat|president|presidency|election|congress|senate|electoral|trump|biden|clinton|aoc|vote|ballot|fake news|administration|liberal|conservativ|socialis|fascis|communis", tsne_dt$text, ignore.case=TRUE))
tsne_dt$house <- as.numeric(grepl("property|homeowner|realtor|real estate|zillow|homebuyer|appraisal|condo|trulia|realty|lender|auction|square footage|sqft|equity|refinanc|house|escrow|mortgage|landlord|foreclos|beachfront", tsne_dt$text, ignore.case=TRUE))
tsne_dt$sciencewords <- as.numeric(grepl("scientist|science|research|hydrogen|atom|molecul|medicine|chemist|astro|study|biolog|geolog|neuro|disease|enzyme|organism|physic|centigrade|cancer", tsne_dt$text, ignore.case=TRUE))
tsne_dt$sciorpolisci <- as.numeric(grepl("climate|environment|vaccine|doctor|virus", tsne_dt$text, ignore.case=TRUE))
tsne_dt$stocks <- as.numeric(grepl("stock|NASDAQ|invest|ipo|sp500|dow jones|sec|capital|share|bear|bull|revenue|S&P", tsne_dt$text, ignore.case=TRUE))
tsne_dt$places <- as.numeric(grepl("mexico|south america|europe|canada", tsne_dt$text, ignore.case=TRUE))
tsne_dt$travelwords <- as.numeric(grepl("airline|airfare|national park|trip|vacation|travel|passport|flight|spanish|airport|visa|TSA|luggage|suitcase|island|backpack|adventure|baggage|customs", tsne_dt$text, ignore.case=TRUE))
tsne_dt$vgames <- as.numeric(grepl("video game|playstation|xbox|multiplayer|single player|ps5|series x|virtual reality|vr|ps4|ps2|ps3|kirby|pokemon|pikachu|rocket league|super smash bros|battle royale|fortnite|among us|call of duty|cod|warzone|modern warfare|league of legends|dota|counter-strike|csgo|world of warcraft|overwatch|zelda|minecraft|nintendo|skyrim|pubg|cyberpunk|red dead|assassin's creed|animal crossing|mario", tsne_dt$text, ignore.case=TRUE))

# Remove text variable again
tsne_dt$text <- NULL

# Write relevant values to CSV file to bring into modeling script
fwrite(tsne_dt, "tsne_dt.csv")
fwrite(train, "train_engineered.csv")
fwrite(test, "test_engineered.csv")


# Front matter
rm(list=ls())
library(data.table)
library(xgboost)

# Read in feature engineered data
data <- fread("tsne_dt.csv")

# Split data back into train and test
train <- data[tt == 0]
# Save and remove ID variables
train_id <- train$id
train$id <- NULL
train_response <- train$response
test <- data[tt == 1]
test_id <- test$id
test$id <- NULL
test_response <- test$response

# Transform data to matrices for XGBoost
train <- as.matrix(train)
test <- as.matrix(test)

# Now explore XGBoost
boosttrain <- xgb.DMatrix(train, label=train_response, missing=NA)
boosttest <- xgb.DMatrix(test, label=test_response, missing=NA)

# Identify tuning parameters
tuning <- NULL

# Combinations of parameters tested:
# gamma = .002, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .16335
# gamma = .0002, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18357
# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .17745
# gamma = .002, eta = .002, max_depth = 18, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18715
# gamma = .002, eta = .002, max_depth = 22, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .19104
# gamma = .002, eta = .001, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .1866
# gamma = .002, eta = .004, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18967
# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .20224

# gamma = .02, eta = .002, max_depth = 20, min_child_weight = 2, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .244055
# gamma = .002, eta = .002, max_depth = 20, min_child_weight = 2, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .23742
# gamma = .01, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18592
# gamma = .02, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .212437
# gamma = .006, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .21088
# gamma = .01, eta = .01, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .20625
# gamma = .01, eta = .006, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .2014
# gamma = .01, eta = .001, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .180989
# gamma = .01, eta = .002, max_depth = 22, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18549
# gamma = .01, eta = .002, max_depth = 18, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .202382
# gamma = .01, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 0.9, colsample_bytree = 1.0, test-mlogloss = .1846

# gamma = .002, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .2020
# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18123
# gamma = .0009, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .18761

# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .17851
# gamma = .001, eta = .001, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .1963
# gamma = .001, eta = .004, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .20981
# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 2, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .230043

# The best parameters we've found are
# gamma = .001, eta = .002, max_depth = 20, min_child_weight = 1, subsample = 1.0, colsample_bytree = 1.0, test-mlogloss = .17851

param <- list(  objective           = "multi:softprob",
                num_class           = 10,
                gamma               = 0.0001,
                booster             = "gbtree",
                eval_metric         = "mlogloss",
                eta                 = 0.002,
                max_depth           = 20,
                min_child_weight    = 1,
                subsample           = 1.0,
                colsample_bytree    = 1.0,
                tree_method = 'hist')

XGBm <- xgb.cv(params=param, nfold=5, nrounds=10000, data=boosttrain, print_every_n=1, early_stopping_rounds=25, missing=NA)

# Save best results
iter_results <- data.table(t(param), best_iter = XGBm$best_iteration, rmse = XGBm$evaluation_log$test_rmse_mean[XGBm$best_iteration])
tuning <- rbind(tuning, iter_results)
fwrite(tuning, "besttuning.csv")

# Fit the model to training data
watchlist <- list(train = boosttrain)
XGBm <- xgb.train(params=param,nrounds=143,missing=NA,data=boosttrain,watchlist=watchlist,print_every_n=1)

# Generate, format, and save testing predictions & model
pred <- predict(XGBm, newdata = boosttest)
pred <- unlist(pred)
submit <- matrix(pred,ncol=10,byrow=T)
colnames(submit) <- c("subredditcars", "subredditCooking", "subredditMachineLearning", "subredditmagicTCG", "subredditpolitics",	"subredditReal_Estate",	"subredditscience",	"subredditStockMarket",	"subreddittravel", "subredditvideogames")

final <- data.table(test_id)
final$id <- final$test_id
final$test_id <- NULL
final <- cbind(final, submit)

summary(XGBm)
saveRDS(XGBm, "finalboost.model")

fwrite(final, "finalsubmission.csv")

