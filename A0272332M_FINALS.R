setwd("/Users/tech26/Desktop/NUS/ACADEMICS/DSA/DSA1101/Finals")
set.seed(2811)

#q1a)
# FALSE

#q1b)
# FALSE

#q1c)
# FALSE

#q1d)
# FALSE

#q1e)
#FALSE

#q2a)
# A

#q2b)
# B

#q2c)
# D 

#q2d)
# Not Pictured

#q2e)
# C




#q3)
data1 <- read.csv("data1-finals.csv")
dim(data1)
head(data1)
attach(data1)
data1 <- data1[,-1]
data1$T <- as.factor(data1$T)
data1$Y <- as.factor(data1$Y)
table(Y)

M1<- glm( Y ~., data =data1,family = binomial)
summary(M1)
# −1.401 + -0.0682(Duration) -1.666I(T = 1) 
# -1.666 is the coefficient for the observation T = 1 

# There are no regressors not significant at level 0.1

# Coefficient of the variable Duration(D) is 0.0682
# It means, given the same condition on the T(whether a laryngeal mask airway or a tracheal tube is used),
# when duration is increased by 1 minute the LOG-ODDS of a patient feeling sore throat 
# increases by 0.0682.

# Coefficient of the variable whether a laryngeal mask airway or a tracheal tube is used(T) is -1.666
# It means, given the same duration of surgery carried out
# when comparing laryngeal mask airway being used, the LOG-ODDS of patient feeling sore throat  
# when a tracheal tube is reduced by 1.666

pred = predict(M1, type="response") # type = response to get the probability of survived
pred_log = prediction(pred, Y)
roc_log = performance(pred_log, measure="tpr", x.measure="fpr")
plot(roc_log, col = "red")
auc1 = performance(pred_log , measure ="auc")
auc1@y.values[[1]]
# AUC = 0.869

plot(roc_log, col = "red")
threshold <- round(as.numeric(unlist(roc_log@alpha.values)),3)
length(threshold)
fpr <- round(as.numeric(unlist(roc_log@x.values)),3)
tpr <- round(as.numeric(unlist(roc_log@y.values)),3)
par(mar= c(5,5,2,5))
plot(threshold,tpr,xlab ='Threshold',ylab = 'True positive rate', type = 'l', col = 'blue')
par(new='True')
plot(threshold,fpr,xlab = '', ylab = '', axes =F, xlim = c(0,1),type = 'l', col = 'red')

axis(side = 4)
mtext(side=4,line =3, 'False positive rate')
text(0.6,0.3,'FPR')
text(0.5,0.7,'TPR')

tpr
fpr
threshold
#treshold = 0.204 gives best threshold for fpr <= 0.5 and maximum


A <- data.frame(Duration = 80, T = 0)
A$T <- as.factor(A$T)
predict(M1, newdata = A, type = 'response')

B <- data.frame(Duration = 125, T = 1)
B$T <- as.factor(B$T)
predict(M1, newdata = B, type = 'response')

# probability of sore throat of A is 0.983
# probability of sore throat of B is 0.996

library(e1071)
M2 <- naiveBayes(Y ~ .,data1)

results <- predict (M2,data1,"class")
confusion.matrix=table(results, Y); confusion.matrix 
sum(diag(confusion.matrix))/sum(confusion.matrix) 
#accuracy 0.829

A <- data.frame(Duration = 80, T = 0)
A$T <- as.factor(A$T)

B <- data.frame(Duration = 125, T = 1)
B$T <- as.factor(B$T)

predict (M2,A,"raw")
predict (M2,B,"raw")

# probability of sore throat of A is 0.997
# probability of sore throat of B is 1.00(when rounded off)

#install.packages("rpart")
#install.packages("rpart.plot")
library("rpart")
library("rpart.plot")

M3 <- rpart(Y ~.,
             method="class",
             data=data1,
             control=rpart.control(minsplit=4),
             parms=list(split='information')
)

rpart.plot(M3, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)
# according to the plotted decision tree the duration is the most important variable
# more specifically duration less than than 23 will immediately categorise the response 
# to be not having sore throat after the surgery regardless of the observation of the T regressor
# whether a laryngeal mask airway or a tracheal tube is used

results_dec <- predict(M3,newdata=data1,type="class")
confusion.matrix=table(results_dec, Y); confusion.matrix
sum(diag(confusion.matrix))/sum(confusion.matrix) 

# accuracy is 0.914

A <- data.frame(Duration = 80, T = 0)
A$T <- as.factor(A$T)

B <- data.frame(Duration = 125, T = 1)
B$T <- as.factor(B$T)

predict(M3,newdata=A,type="class")
predict(M3,newdata=B,type="class")

# A is predicted to have sore throat after the surgery
# B is predicted to have sore throat after the surgery

#q4)

data2 <- read.csv("data2-finals.csv")
dim(data2)
head(data2)
attach(data2)

plot(data2[,-1])

K = 10
wss <- numeric(K)
for (k in 1:K) { 
  wss[k] <- sum(kmeans(data2[,c("sepal.length","sepal.width","petal.length","petal.width")], centers=k)$withinss)
}
wss

plot(1:K, wss, col = "blue", type="b", xlab="Number of Clusters",  ylab="Within Sum of Squares")
# k = 3 should be chosen as from k = 1 to k = 2 a sharp non linear drop in WSS and from k = 1 to k = 2 a less steep
# non linear drop in WSS is seen
# However, from k = 3 to k = 10 a steady linear drop in WSS is seen. 
# Since we want to balance the lowest possible WSS with a not 
# too big value of K we choose k = 3 so as to ensure the model has a 
# small WSS but not too many clusters. Too many clusters will result in the 
# model being unnccessarilty complicated 

kout <- kmeans(data2[,c("sepal.length","sepal.width","petal.length","petal.width")],centers=3)

plot(data2[,-1], col=kout$cluster)
kout

# centroids of the clusters
# cluster 1: (sepal.length = 5.006, sepal.width = 3.418, petal.length = 1.464 , petal.width = 0.244)
# cluster 2: (sepal.length = 5.902, sepal.width = 2.748, petal.length = 4.394 , petal.width = 1.433)
# cluster 3: (sepal.length = 6.850, sepal.width = 3.074, petal.length = 5.742 , petal.width = 2.071)

# sizes of each cluster 
# cluster 1: 50
# cluster 2: 62
# cluster 3: 38






