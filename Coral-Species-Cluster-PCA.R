# Coral Species Unsupervised Learning
# Author: Jess Strait

# Clear environment and load packages
rm(list = ls())
library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ClusterR)
library(cluster)
library(mlr)

# Intake data
data <- fread("speciesdata.csv")

# Save IDs
id_vector <- data$id
data$id <- NULL

# Convert values to factors for dummies
data$locus_1 <- as.factor(data$locus_1)
data$locus_2 <- as.factor(data$locus_2)
data$locus_3 <- as.factor(data$locus_3)
data$locus_4 <- as.factor(data$locus_4)
data$locus_5 <- as.factor(data$locus_5)
data$locus_6 <- as.factor(data$locus_6)
data$locus_7 <- as.factor(data$locus_7)
data$locus_8 <- as.factor(data$locus_8)
data$locus_9 <- as.factor(data$locus_9)
data$locus_10 <- as.factor(data$locus_10)
data$locus_11 <- as.factor(data$locus_11)
data$locus_12 <- as.factor(data$locus_12)
data$locus_13 <- as.factor(data$locus_13)
data$locus_14 <- as.factor(data$locus_14)
data$locus_15 <- as.factor(data$locus_15)

# create dummies variables
dummies <- dummyVars(~ ., data = data)
numdummies <- predict(dummies, newdata = data)

# Run a principal component analysis
pca <- prcomp(numdummies)
screeplot(pca)
summary(pca)
biplot(pca)

# Save principal component coordinates
pca_dt <- data.table(pca$x)

# Kmeans clustering with PC's - best performance is with all PC's
kmean_sol <- kmeans(pca_dt[,], centers = 3, nstart = 25)
pca_dt$kmeanPred <- kmean_sol$cluster

# Save kmeans model
saveRDS(kmean_sol, "kmeans.model")

# Add back the ID values for Phase 1 submission
#submission <- data.table(pca_dt$kmeanPred)
#submission$id <- id_vector

# Generate submission file for Phase 1 submission
#submission$species1 <- 0.33
#submission$species1[grep('1', submission$V1)] <- 0.66
#submission$species2 <- 0.33
#submission$species2[grep('2', submission$V1)] <- 0.66
#submission$species3 <- 0.33
#submission$species3[grep('3', submission$V1)] <- 0.66
#submission$V1 <- NULL

# Phase 2 code start

# Remove kmeans information
pca_dt$kmeanPred <- NULL

# Run tSNE baseline as shown in class
set.seed(3)
perplexityvalue <- 10
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)

# Obtain tSNE coordinates and observe clustering
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

# Test other perplexity values
perplexityvalue <- 30
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

perplexityvalue <- 50
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

perplexityvalue <- 20
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

perplexityvalue <- 15
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

# 30 was an improvement from 10. 50 was not an improvement from 30. 20 was an improvement from 30. 15 was worse than 20. Proceed with 20.
# 1000 iterations did not beat the benchmark. Try more iterations.

perplexityvalue <- 20
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F, max_iter = 10000)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

# Okay, so more iterations made the model worse. Try less iteration (last submission).

perplexityvalue <- 20
tsne <- Rtsne(pca_dt, pca = F, perplexityvalue=perplexityvalue, check_duplicates = F, max_iter = 300)
tsne_dt <- data.table(tsne$Y)
ggplot(tsne_dt, aes(x=V1, y=V2)) + geom_point() + labs(title = paste("perplexity = ", perplexityvalue))

# We know from the competition that the optimal number of clusters is k=3
gmm_data <- GMM(tsne_dt[,.(V1,V2)], 3)

# Convert log-likelihood into probability as shown in lecture (remember that likelihood and probability are different)
l_clust <- gmm_data$Log_likelihood^10
l_clust <- data.table(l_clust)
net_lh <- apply(l_clust,1,FUN=function(x){sum(1/x)})
cluster_prob <- 1/l_clust/net_lh

# Observe cluster 1 probabilities
tsne_dt$Cluster_1_prob <- cluster_prob$V1
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_1_prob)) + geom_point()

# Observe cluster 2 probabilities
tsne_dt$Cluster_2_prob <- cluster_prob$V2
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_2_prob)) + geom_point()

# Observe cluster 3 probabilities
tsne_dt$Cluster_3_prob <- cluster_prob$V3
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_3_prob)) + geom_point()

# Assign labels
tsne_dt$gmm_labels <- max.col(cluster_prob, ties.method = "random")
ggplot(tsne_dt,aes(x=V1,y=V2,col=gmm_labels)) + geom_point()

# Create submission file
tsne_dt$id <- id_vector
submissionfinal <- tsne_dt
submissionfinal$V1 <- NULL
submissionfinal$V2 <- NULL
submissionfinal$species1 <- submissionfinal$Cluster_3_prob
submissionfinal$species2 <- submissionfinal$Cluster_2_prob
submissionfinal$species3 <- submissionfinal$Cluster_1_prob
submissionfinal$Cluster_3_prob <- NULL
submissionfinal$Cluster_2_prob <- NULL
submissionfinal$Cluster_1_prob <- NULL
submissionfinal$gmm_labels <- NULL

fwrite(submissionfinal, "tsne_submission_20_less_iters.csv")


