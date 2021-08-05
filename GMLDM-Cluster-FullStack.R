# GMLDM/MEQ Cluster Analysis
# Author: Jess Strait for CROPP Cooperative

# Clear environment and load packages
rm(list = ls())
setwd("~/")
library(data.table)
library(Rtsne)
library(ggplot2)
library(caret)
library(ClusterR)
library(cluster)
library(mlr)
library(factoextra)
library(NbClust)
library(FeatureImpCluster)
library(attempt)
library(flexclust)
library(RColorBrewer)

# Intake data
data <- fread("ChannelMEQ.csv")
data <- na.omit(data)

# Drop outliers seen in analysis
data <- data %>% filter(`Milk Equivalents` < 50000000)
data <- data %>% filter(`Discounts & Allowances` > -1500000)
data <- data %>% filter(`GMLDM $` < 2000000)

# Create dummy variables
dummies <- dummyVars(~ ., data = data)
numdummies <- predict(dummies, newdata = data)

# Run tSNE
set.seed(3)
perplexityvalue <- 50
tsne <- Rtsne(numdummies, pca = F, perplexityvalue=perplexityvalue, theta = 0.4, check_duplicates = F)

# Obtain tSNE coordinates and visually observe clustering
tsne_dt <- data.table(tsne$Y)

# Elbow method
fviz_nbclust(tsne_dt, kmeans, method = "wss", k.max=25) +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
# Suggests optimal number of clusters is 4

# Silhouette method
fviz_nbclust(tsne_dt, kmeans, method = "silhouette", k.max=25)+
  labs(subtitle = "Silhouette method")
# Suggests optimal number of clusters is 3

# Run kmeans with tSNE dataframe
kmeantsne <- kmeans(tsne_dt, centers = 4, nstart = 100, algorithm=c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"), trace=FALSE)
# KCCA family object for feature importance vis
res <- kcca(tsne_dt, family=kccaFamily("kmeans"), k=4)


# Visualize the clusters in color
fviz_cluster(kmeantsne, tsne_dt,  palette = c("pink1", "violet", "mediumpurple1", "slateblue1", "purple", "forestgreen",
                                              "turquoise2", "skyblue", "yellow", "blue2", "navyblue",
                                              "orange", "tomato", "black", "palevioletred", "violetred", "red2",
                                              "springgreen2", "yellowgreen", "palegreen4",
                                              "wheat2", "tan3", "brown3",
                                              "grey70"), ggtheme = theme_minimal())

# Save cluster predictions as dataframe columns
tsne_dt$kmeanPred <- kmeantsne$cluster
data$kmeanPred <- as.factor(kmeantsne$cluster)

# Compute group means
commeans <- data %>% group_by(Commodity) %>% mutate(`GMLDM $` = mean(`GMLDM $`))
commeans <- commeans %>% select(`GMLDM $`, `Milk Equivalents`, `Commodity`, `kmeanPred`)
commeans$kmeanPred <- 0

chanmeans <- data %>% group_by(`Sales Channel`) %>% mutate(`GMLDM $` = mean(`GMLDM $`))
chanmeans <- chanmeans %>% select(`GMLDM $`, `Milk Equivalents`, `kmeanPred`, `Sales Channel`)
chanmeans$kmeanPred <- 0

ggplot(data, aes(x=`Milk Equivalents`, y=`Discounts & Allowances`, color=kmeanPred)) + geom_point()
# Clust 2: Higher milk equivalents; no clear D&A
# Clust 1: Midrange milk equivalents; no clear D&A
# Clust 4: Low milk equivalents; no clear D&A
# Clust 3: Very low milk equivalents; almost zero D&A
ggplot(data, aes(x=`Milk Equivalents`, y=`GMLDM $`, color=kmeanPred)) + geom_point()
# Clust 4 and somewhat 3: NEGATIVE GMLDM
ggplot(data, aes(x=`GMLDM $`, y=Commodity, color=kmeanPred)) + geom_point()
# Clust 2: mostly bulk milk with high GMLDM; a couple UHT, spreads, powder (range), HTST (high GMLDM), cheese
# Clust 4: mostly sour cream, butter, aseptic (where we see low GMLDM)
# Clust 3: UHT, powder, cheese- all almost 0 GMLDM
ggplot(data, aes(x=`Discounts & Allowances`, y=Commodity, color=kmeanPred)) + geom_point()
ggplot(data, aes(x=`Milk Equivalents`, y=Commodity, color=kmeanPred)) + geom_point()
ggplot(data, aes(x=`GMLDM $`, y=`Sales Channel`, color=kmeanPred)) + geom_point()
ggplot(data, aes(x=`Discounts & Allowances`, y=`Sales Channel`, color=kmeanPred)) + geom_point()
ggplot(data, aes(x=`Milk Equivalents`, y=`Sales Channel`, color=kmeanPred)) + geom_point()
# Clust 4: manufacturing ops division, food service division, global ingredients; all low GMLDM
# Clust 3: small format
# Clust 1: Retail grocery, ingredient, food service
# Clust 2: Mostly retailer grocery, some ingredient and high GMLDM VP
# Clust 3: VP and Sales Operations
# Clust 4: Mostly VP and some ingredient
# Manufacturing, small format, VP divisions -> all 0 GMLDM
ggplot(data, aes(x=`GMLDM $`, y=reorder(`Commodity`, `GMLDM $`), color=kmeanPred, size=`Milk Equivalents`)) + geom_point() + geom_point(data=commeans, aes(color='Mean GMLDM', size=100)) + ylab("Commodity")
ggplot(data, aes(x=`GMLDM $`, y=reorder(`Sales Channel`, `GMLDM $`), color=kmeanPred, size=`Milk Equivalents`)) + geom_point() + geom_point(data=chanmeans, aes(color='Mean GMLDM', size=300)) + ylab("Sales Channel")

ggplot(data, aes(x=`Milk Equivalents`, y=`GMLDM $`, color=Commodity)) + geom_point()
ggplot(data, aes(x=`Milk Equivalents`, y=`GMLDM $`, color=`Sales Channel`)) + geom_point()


# Divs with higher GMLDM: domestic ingredient division
# Cmdty with higher GMLDM: Bulk milk, powder
# Cmdty with lower GMLDM: Spreads, sour cream, cottage cheese, some butter

gmm_data <- GMM(tsne_dt[,.(V1,V2)], 4)

# Convert log-likelihood into probability of cluster occurrence
l_clust <- gmm_data$Log_likelihood^10
l_clust <- data.table(l_clust)
net_lh <- apply(l_clust,1,FUN=function(x){sum(1/x)})
cluster_prob <- 1/l_clust/net_lh

tsne_dt$Cluster_1_prob <- cluster_prob$V1
tsne_dt$Cluster_2_prob <- cluster_prob$V2
tsne_dt$Cluster_3_prob <- cluster_prob$V3
tsne_dt$Cluster_4_prob <- cluster_prob$V4

# Visualize cluster probabilities - did not find this method useful
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_1_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_2_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_3_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_4_prob)) + geom_point()

# Write data with cluster predictions to a CSV
write.csv(data, "TsneProductFourClusterAnalysis.csv")

# Clustering for meat & egg pools
rm(list = ls())
meat <- fread("Meat&amp;Egg.csv")
meat <- na.omit(meat)
meat <- meat %>% filter(`GMLDM $` < 100000)
meatdummies <- dummyVars(~ ., data = meat)
meatnumdummies <- predict(meatdummies, newdata = meat)

# Run tSNE
set.seed(3)
perplexityvalue <- 50
tsne <- Rtsne(meatnumdummies, pca = F, perplexityvalue=perplexityvalue, theta = 0.4, check_duplicates = F)

# Obtain tSNE coordinates and visually observe clustering
tsne_dt <- data.table(tsne$Y)

# Run kmeans again with tSNE dataframe
kmeantsne <- kmeans(tsne_dt, centers = 4, nstart = 100, algorithm=c("Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"), trace=FALSE)
# KCCA family object for feature importance vis
res <- kcca(tsne_dt, family=kccaFamily("kmeans"), k=4)

# Elbow method
fviz_nbclust(tsne_dt, kmeans, method = "wss", k.max=25) +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")
# Suggests optimal number of clusters is 4

# Silhouette method
fviz_nbclust(tsne_dt, kmeans, method = "silhouette", k.max=25)+
  labs(subtitle = "Silhouette method")
# Suggests optimal number of clusters is 2

# Visualize the clusters in color
fviz_cluster(kmeantsne, tsne_dt,  palette = c("pink1", "violet", "mediumpurple1", "slateblue1", "purple", "forestgreen",
                                              "turquoise2", "skyblue", "yellow", "blue2", "navyblue",
                                              "orange", "tomato", "black", "palevioletred", "violetred", "red2",
                                              "springgreen2", "yellowgreen", "palegreen4",
                                              "wheat2", "tan3", "brown3",
                                              "grey70"), ggtheme = theme_minimal())

# Save cluster predictions as dataframe columns
tsne_dt$kmeanPred <- kmeantsne$cluster
meat$kmeanPred <- as.factor(kmeantsne$cluster)

# Compute group means
commeans <- meat %>% group_by(Commodity) %>% mutate(`GMLDM $` = mean(`GMLDM $`))
commeans <- commeans %>% select(`GMLDM $`, `Commodity`, `kmeanPred`)
commeans$kmeanPred <- 0

chanmeans <- meat %>% group_by(`Sales Channel`) %>% mutate(`GMLDM $` = mean(`GMLDM $`))
chanmeans <- chanmeans %>% select(`GMLDM $`, `kmeanPred`, `Sales Channel`)
chanmeans$kmeanPred <- 0

ggplot(meat, aes(x=`GMLDM $`, y=Commodity, color=kmeanPred)) + geom_point()
# Clust 4: Ranging all commodities, both high and low outliers- most eggs
# Eggs have higher average GMLDM
# Chicken and turkey tend towards zero GMLDM
# Clust 1: Almost zero GMLDM
# Clust 3: Exactly zero
ggplot(meat, aes(x=`Discounts & Allowances`, y=Commodity, color=kmeanPred)) + geom_point()
# Clust 4: Has almost all D&A, all other clusters tend towards zero
ggplot(meat, aes(x=`GMLDM $`, y=`Sales Channel`, color=kmeanPred)) + geom_point()
# Retail grocery has trending positive GMLDM
# Ingredient and VP less extreme but also trending positive
# Sales Operations exactly zero, foodservice trending towards zero
ggplot(meat, aes(x=`Discounts & Allowances`, y=`Sales Channel`, color=kmeanPred)) + geom_point()
# Retail grocery has almost all D&A, with some VP
ggplot(meat, aes(x=`GMLDM $`, y=reorder(`Commodity`, `GMLDM $`), color=kmeanPred)) + geom_point() + geom_point(data=commeans, aes(color='Mean GMLDM')) + ylab("Commodity")
ggplot(meat, aes(x=`GMLDM $`, y=reorder(`Sales Channel`, `GMLDM $`), color=kmeanPred)) + geom_point() + geom_point(data=chanmeans, aes(color='Mean GMLDM')) + ylab("Sales Channel")

gmm_data <- GMM(tsne_dt[,.(V1,V2)], 4)

# Convert log-likelihood into probability of cluster occurrence
l_clust <- gmm_data$Log_likelihood^10
l_clust <- data.table(l_clust)
net_lh <- apply(l_clust,1,FUN=function(x){sum(1/x)})
cluster_prob <- 1/l_clust/net_lh

tsne_dt$Cluster_1_prob <- cluster_prob$V1
tsne_dt$Cluster_2_prob <- cluster_prob$V2
tsne_dt$Cluster_3_prob <- cluster_prob$V3
tsne_dt$Cluster_4_prob <- cluster_prob$V4

# Visualize cluster probabilities - did not find this method useful
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_1_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_2_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_3_prob)) + geom_point()
ggplot(tsne_dt,aes(x=V1,y=V2,col=Cluster_4_prob)) + geom_point()


# Write data with cluster predictions to a CSV
write.csv(data, "TsneMeatFourClusterAnalysis.csv")
