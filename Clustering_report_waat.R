library(readr)
library(ggplot2)
library(ggrepel)
library(tidyverse)
library(dplyr) 
library(e1071)
library(pcaMethods)
library(FactoMineR) 

dataset_finale <- read_csv("~/PycharmProjects/00192_00194_grosso_fercia/dataset_finale.csv", na= 'None')
ds <- dataset_finale %>% mutate_all(~ifelse(is.na(.x), mean(.x, na.rm = TRUE), .x)) 
ds$`Risultato Investimento` <-factor(dataset_finale$`Risultato Investimento`,c('Negative','Positive'),c(0,1))
view(dataset_finale)

ds<-dataset_finale[-17]
view(ds)

ds_cluster <- ds[c(1,4,5)]
view(ds_cluster)
attach(ds_cluster)

################################################################################
## KMEANS CLUSTERING (K=2): Investimenti_Green and Tweet social media
################################################################################

km.out=kmeans(ds_cluster[-1],2,nstart=20)
km.out$cluster
plot(ds_cluster[-1], col=(km.out$cluster+1), main="K-Means Clustering Results with K=2", xlab="", ylab="", pch=20, cex=2,)
text(ds_cluster, labels=ds_cluster$Country)

gg3<-ggplot(ds_cluster, aes(x=`Investimenti Green`, y=Total, color = (km.out$cluster))) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('Kmeans Clustering Investimenti green e social media')
gg3

################################################################################
## KMEANS CLUSTERING (K=3)
################################################################################

set.seed(4)
km.out2=kmeans(ds_cluster[-1],3,nstart=20)
km.out2
plot(ds_cluster[-1], col=(km.out2$cluster+1), main="K-Means Clustering Results with K=2", xlab="", ylab="", pch=20, cex=2,)
km.out2$tot.withinss

gg4<-ggplot(ds_cluster, aes(x=`Investimenti Green`, y=Total, color = (km.out2$cluster+1))) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('K-Means Clustering Results with K=3')
gg4

ds_cluster <- ds[c(1,4,8)]
view(ds_cluster)
attach(ds_cluster)
km.out=kmeans(ds_cluster[-1],2,nstart=20)
km.out$cluster
plot(ds_cluster[-1], col=(km.out$cluster+1), main="K-Means Clustering Results with K=2", xlab="", ylab="", pch=20, cex=2,)
text(ds_cluster, labels=ds_cluster$Country)

gg3<-ggplot(ds_cluster, aes(x=Total, y=`RE_%_of_total`, color = (km.out$cluster))) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('Kmeans Clustering Investimenti green e social media')
gg3

################################################################################
## KMEANS CLUSTERING (K=2): Investimenti_green and Population
################################################################################

ds_cluster <- ds[c(1,8,10)]
view(ds_cluster)
km.out3=kmeans(ds_cluster[-1],3,nstart=20)
km.out3$cluster
plot(ds_cluster[-1], col=(km.out3$cluster+1), main="K-Means Clustering Results with K=2", xlab="Tweet", ylab="Population", pch=20, cex=2,)
text(ds_cluster, labels=ds_cluster$Country)

gg5<-ggplot(ds_cluster, aes(x=`Population`, y=Total, color = (km.out3$cluster))) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('K-Means Clustering on Population and Total tweet with K=3') 
gg5

ds_cluster <- ds[c(1,4,13)]
attach(ds_cluster)
view(ds_cluster)
km.out3=kmeans(ds_cluster[-1],2,nstart=50)
km.out3$cluster
plot(ds_cluster[-1], col=(km.out3$cluster+1), main="K-Means Clustering Results with K=2", xlab="Tweet", ylab="Population", pch=20, cex=2,)
text(ds_cluster, labels=ds_cluster$Country)

gg6<-ggplot(ds_cluster, aes(x=`RE_%_of_total`, y=`Fossil CO2 Emissions: 2017vs1990, change(%)`, color = (km.out3$cluster))) + geom_point() + 
  geom_label_repel(aes(label = Country),
                   box.padding   = 0.35, 
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  theme_classic() + ggtitle('K-Means Clustering on Fossil CO2 Emissions: 2017vs1990, change(%) and RE_%_of_total with K=2') 
gg6

################################################################################
## HIERARCHICAL CLUSTERING
################################################################################

hc.complete=hclust(dist(ds[-1]), method="complete")
hc.average=hclust(dist(ds[-1]), method="average")
hc.single=hclust(dist(ds[-1]), method="single")
par(mfrow=c(1,3))
plot(hc.complete,main="Complete Linkage", xlab="Country", cex=.9)
plot(hc.average, main="Average Linkage", xlab="Country", cex=.9)
plot(hc.single, main="Single Linkage", xlab="Country", cex=.9)
cutree(hc.complete, 2)
cutree(hc.average, 2)
cutree(hc.single, 2)

xsc=scale(ds[-1])
plot(hclust(dist(xsc), method="complete"), main="Hierarchical Clustering with Scaled Features Complete")
plot(hclust(dist(xsc), method="average"), main="Hierarchical Clustering with Scaled Features Average Linkage")
plot(hclust(dist(xsc), method="single"), main="Hierarchical Clustering with Scaled Features Single Linkage")

par(mfrow=c(1,1))
################################################################################
## PRINCIPAL COMPONENT ANALYSIS
################################################################################

res <- PCA(ds,quali.sup=c(1),quanti.sup = c(16,15,14,13),scale.unit = TRUE, ncp=18)
summary(res)

plot(res, choix="var", title="Graph of the variables", axes=1:2)
plot(res, choix="var", title="Graph of the variables", axes=3:4)

dimdesc(res)
plot(res, cex=0.8, invisible="quali", title="Graph of the individuals")


#Eigenvalues
#                       Dim.1   Dim.2   Dim.3   Dim.4   Dim.5   Dim.6   Dim.7   Dim.8   Dim.9
#Variance               5.905   4.053   1.350   1.015   0.433   0.149   0.077   0.009   0.003
#% of var.             45.424  31.180  10.386   7.810   3.331   1.145   0.593   0.070   0.026
#Cumulative % of var.  45.424  76.603  86.990  94.799  98.130  99.275  99.868  99.938  99.963

################################################################################
## PRINCIPAL COMPONENT ANALYSIS: prcomp
################################################################################

PCA_2 <- prcomp(dataset_finale[-c(17,1)], scale. = T, center = T) #principal c regression
summary(PCA_2)

plot(PCA_2$x[,1:2], col = dataset_finale$`Risultato Investimento`)
plot(PCA_2$x[,2:3], col = dataset_finale$`Risultato Investimento`)
PCA_2$x 

#Importance of components:
#                         PC1    PC2    PC3     PC4     PC5     PC6     PC7     PC8    PC9    PC10    PC11
#Standard deviation     2.4310 2.0293 1.3625 1.12641 1.08370 0.99307 0.81191 0.71606 0.5577 0.35363 0.25288
#Proportion of Variance 0.3476 0.2422 0.1092 0.07464 0.06908 0.05801 0.03878 0.03016 0.0183 0.00736 0.00376
#Cumulative Proportion  0.3476 0.5899 0.6991 0.77369 0.84278 0.90079 0.93957 0.96973 0.9880 0.99538 0.99914

################################################################################
## CLUSTERING WITH PC
################################################################################

scores = PCA_2$x[]

z1 = scores[,1]
z2 = scores[,2]
z3 = scores[,3]    
z4 = scores[,4]    

## HIERARCHICAL WITH PC
hc = hclust(dist(cbind(z1,z2,z3,z4)), method = 'ward.D')
plot(hc, axes=F,xlab='', ylab='',sub ='', 
     main='hClust Comp 1-4')
rect.hclust(hc, k=3, border='red')
