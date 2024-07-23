set.seed(1101)
setwd("/Users/tech26/Desktop/NUS/ACADEMICS/DSA/DSA1101/DSA assignment")
diabetes <- read.csv("diabetes_5050.csv")
attach(diabetes)
diabetes$Diabetes_binary = as.factor(diabetes$Diabetes_binary)
diabetes$HighBP = as.factor(diabetes$HighBP)
diabetes$HighChol = as.factor(diabetes$HighChol)
diabetes$CholCheck = as.factor(diabetes$CholCheck)
diabetes$Stroke = as.factor(diabetes$Stroke)
diabetes$HeartDiseaseorAttack = as.factor(diabetes$HeartDiseaseorAttack)
diabetes$Veggies = as.factor(diabetes$Veggies)
diabetes$HvyAlcoholConsump = as.factor(diabetes$HvyAlcoholConsump)
diabetes$DiffWalk = as.factor(diabetes$DiffWalk)




### ANALYSING DATA SET
dim(diabetes)
head(diabetes)
summary(diabetes) 
# min, 1st, median, mean, mean, 3rd, max of each column
table(Diabetes_binary)
table(Stroke)
table(BMI)

### ASSOCIATION BW VARIBLES
# Perform logistic regression to identify varibales that have a strong association with the response variable
# i.e important variable in generating classifiers

variables = diabetes[,-1]
Test_model <- glm( Diabetes_binary ~., data =variables,family = binomial)
summary(Test_model)

# Based on the p value
# Reject variables "Smoker", "PhysActivity", "Fruits", "AnyHealthcare", "NoDocbcCost" 
# Do not have a strong association to  the response Diabetes_binary

# Perform Odds Ratio Calculation
table_HighBP = table(HighBP, Diabetes_binary);table_HighBP
tab_HighBP = prop.table(table_HighBP, "HighBP");tab_HighBP
OR_HighBP = (tab_HighBP[1]/(1-tab_HighBP[1]))/(tab_HighBP[2]/(1-tab_HighBP[2])); OR_HighBP

table_HighChol = table(HighChol, Diabetes_binary);table_HighChol
tab_HighChol = prop.table(table_HighChol, "HighChol");tab_HighChol
OR_HighChol = (tab_HighChol[1]/(1-tab_HighChol[1]))/(tab_HighChol[2]/(1-tab_HighChol[2])); OR_HighChol

table_CholCheck = table(CholCheck, Diabetes_binary);table_CholCheck
tab_CholCheck = prop.table(table_CholCheck, "CholCheck");tab_CholCheck
OR_CholCheck = (tab_CholCheck[1]/(1-tab_CholCheck[1]))/(tab_CholCheck[2]/(1-tab_CholCheck[2])); OR_CholCheck

table_BMI = table(BMI, Diabetes_binary);table_BMI
tab_BMI = prop.table(table_BMI, "BMI");tab_BMI
OR_BMI = (tab_BMI[1]/(1-tab_BMI[1]))/(tab_BMI[2]/(1-tab_BMI[2])); OR_BMI

table_Stroke = table(Stroke, Diabetes_binary);table_Stroke
tab_Stroke = prop.table(table_Stroke, "Stroke");tab_Stroke
OR_Stroke = (tab_Stroke[1]/(1-tab_Stroke[1]))/(tab_Stroke[2]/(1-tab_Stroke[2])); OR_Stroke

table_HeartDiseaseorAttack = table(HeartDiseaseorAttack, Diabetes_binary);table_HeartDiseaseorAttack
tab_HeartDiseaseorAttack = prop.table(table_HeartDiseaseorAttack, "HeartDiseaseorAttack");tab_HeartDiseaseorAttack
OR_HeartDiseaseorAttack = (tab_HeartDiseaseorAttack[1]/(1-tab_HeartDiseaseorAttack[1]))/(tab_HeartDiseaseorAttack[2]/(1-tab_HeartDiseaseorAttack[2])); OR_HeartDiseaseorAttack

table_Veggies = table(Veggies, Diabetes_binary);table_Veggies
tab_Veggies = prop.table(table_Veggies, "Veggies");tab_Veggies
OR_Veggies = (tab_Veggies[1]/(1-tab_Veggies[1]))/(tab_Veggies[2]/(1-tab_Veggies[2])); OR_Veggies

table_HvyAlcoholConsump = table(HvyAlcoholConsump, Diabetes_binary);table_HvyAlcoholConsump
tab_HvyAlcoholConsump = prop.table(table_HvyAlcoholConsump, "HvyAlcoholConsump");tab_HvyAlcoholConsump
OR_HvyAlcoholConsump = (tab_HvyAlcoholConsump[1]/(1-tab_HvyAlcoholConsump[1]))/(tab_HvyAlcoholConsump[2]/(1-tab_HvyAlcoholConsump[2])); OR_HvyAlcoholConsump

table_GenHlth = table(GenHlth, Diabetes_binary);table_GenHlth
tab_GenHlth = prop.table(table_GenHlth, "GenHlth");tab_GenHlth
OR_GenHlth = (tab_GenHlth[1]/(1-tab_GenHlth[1]))/(tab_GenHlth[2]/(1-tab_GenHlth[2])); OR_GenHlth

table_MentHlth = table(MentHlth, Diabetes_binary);table_MentHlth
tab_MentHlth = prop.table(table_MentHlth, "MentHlth");tab_MentHlth
OR_MentHlth = (tab_MentHlth[1]/(1-tab_MentHlth[1]))/(tab_MentHlth[2]/(1-tab_MentHlth[2])); OR_MentHlth

table_PhysHlth = table(PhysHlth, Diabetes_binary);table_PhysHlth
tab_PhysHlth = prop.table(table_PhysHlth, "PhysHlth");tab_PhysHlth
OR_PhysHlth = (tab_PhysHlth[1]/(1-tab_PhysHlth[1]))/(tab_PhysHlth[2]/(1-tab_PhysHlth[2])); OR_PhysHlth

table_DiffWalk = table(DiffWalk, Diabetes_binary);table_DiffWalk
tab_DiffWalk = prop.table(table_DiffWalk, "DiffWalk");tab_DiffWalk
OR_DiffWalk = (tab_DiffWalk[1]/(1-tab_DiffWalk[1]))/(tab_DiffWalk[2]/(1-tab_DiffWalk[2])); OR_DiffWalk

table_Sex = table(Sex, Diabetes_binary);table_Sex
tab_Sex = prop.table(table_Sex, "Sex");tab_Sex
OR_Sex = (tab_Sex[1]/(1-tab_Sex[1]))/(tab_Sex[2]/(1-tab_Sex[2])); OR_Sex

table_Age = table(Age, Diabetes_binary);table_Age
tab_Age = prop.table(table_Age, "Age");tab_Age
OR_Age = (tab_Age[1]/(1-tab_Age[1]))/(tab_Age[2]/(1-tab_Age[2])); OR_Age

table_Education = table(Education, Diabetes_binary);table_Education
tab_Education = prop.table(table_Education, "Education");tab_Education
OR_Education = (tab_Education[1]/(1-tab_Education[1]))/(tab_Education[2]/(1-tab_Education[2])); OR_Education

table_Income = table(Income, Diabetes_binary);table_Income
tab_Income = prop.table(table_Income, "Income");tab_Income
OR_Income = (tab_Income[1]/(1-tab_Income[1]))/(tab_Income[2]/(1-tab_Income[2])); OR_Income

# Based on odds ratio of the variables, we reject the variables of odds ratio 0.8-1.2 
# Variables  "PhysHlth", "Sex", "Income"as these have a weak association with the reponse variable

# In Totality the variables "Smoker", "PhysActivity", "Fruits", "AnyHealthcare", "NoDocbcCost", "Income", "PhysHlth", "Sex" are removed.
drops<-c("Smoker","PhysActivity","Fruits","AnyHealthcare","NoDocbcCost","PhysHlth","Sex","Income")
diabetes <- diabetes[,!(names(diabetes) %in% drops)]
head(diabetes)

### SPLITTING DATA INTO TRAIN AND TEST SETS
# test data : train data <- 1:4
# test and train data will have equal percentage of positive and negative response
n_folds = 5
folds_negative <- sample(rep(1:n_folds, length.out = dim(diabetes)[1]/2 )) 
folds_positive <- sample(rep(1:n_folds, length.out = dim(diabetes)[1]/2 )) 

table(folds_negative)
table(folds_positive)

negative_data = diabetes[1:35346,] # Only negtaive resposne data
positive_data = diabetes[35347:70692,] # Only positive response data

### DECISION TREE CLASSIFIER

acc_1 = numeric(n_folds)
err_1 = numeric(n_folds)
fpr_values_1 = numeric(n_folds)
fnr_values_1 = numeric(n_folds)

library(rpart)
for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train = rbind(train_nej, train_pos)
  test = rbind(test_nej, test_pos)
  
  model_decision_tree <- rpart(Diabetes_binary ~ ., 
                               method = "class", data =train, control = rpart.control( minsplit =1),
                               parms = list( split ='information'))
  pred = predict(model_decision_tree, newdata = test[,2:14], type = 'class')
  confusion.matrix = table(pred, test[,1])
  
  TP <- confusion.matrix[2, 2]
  TN <- confusion.matrix[1, 1]
  FP <- confusion.matrix[1, 2]
  FN <- confusion.matrix[2, 1]
  fpr_values_1[j] <- FP / (FP + TN)
  fnr_values_1[j] <- FN / (FN + TP)
  
  acc_1[j] = sum(diag(confusion.matrix))/sum(confusion.matrix)
  err_1[j] = 1 - sum(diag(confusion.matrix))/sum(confusion.matrix)
  
  
}
acc_decision_tree = mean(acc_1); acc_decision_tree # accuracy rate
err_decision_tree = mean(err_1); err_decision_tree # error rate
fpr_values_decision_tree = mean(fpr_values_1); fpr_values_decision_tree # FPR rate
fnr_values_decision_tree = mean(fnr_values_1); fnr_values_decision_tree # FPR rate


library(ROCR)
has_diabetes <- predict(model_decision_tree, diabetes[,2:14], type='class')
has_diabetes = as.numeric(paste(has_diabetes))
pred_dt = prediction(has_diabetes, Diabetes_binary)
roc_dt = performance(pred_dt, measure="tpr", x.measure="fpr")
plot(roc_dt) 
auc_dc = performance(pred_dt , measure ="auc")
auc_dc@y.values[[1]] # AUC value

### KNN CLASSIFIER

acc_2 = numeric(n_folds)
err_2 = numeric(n_folds)
fpr_values_2 = numeric(n_folds)
fnr_values_2 = numeric(n_folds)

library(class)
# Doing knn from 1 to 100
K = 100

accuracy = numeric(K)

random <- 5
test_nej <- which(folds_negative == random)
test_pos <- which(folds_positive == random)

train_nej = negative_data[ -test_nej, ]
train_pos = positive_data[ -test_pos, ]
test_nej = negative_data[test_nej, ]
test_pos = positive_data[test_pos, ]

train_knn = rbind(train_nej, train_pos)
test_knn = rbind(test_nej, test_pos)

# We do the n folds loop outside of the K loop so as to ensrue the program runs in a reasonable amount of time
for (i in 1:K) {
  pred <- knn(train = train_knn[, 2:14], test = test_knn[, 2:14], cl = train_knn[,1], k = i)
  accuracy[i]= mean(test_knn[,1] == pred)
}
which(accuracy == max(accuracy))

# The model that gves the best accuracy is k = 88

for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train_knn = rbind(train_nej, train_pos)
  test_knn = rbind(test_nej, test_pos)
  
  model_knn <- knn(train = train_knn[, 2:14], test = test_knn[, 2:14], cl = train_knn$Diabetes_binary, k = 88)
  confusion.matrix = table(model_knn, test_knn[,1])
  
  TP <- confusion.matrix[2, 2]
  TN <- confusion.matrix[1, 1]
  FP <- confusion.matrix[1, 2]
  FN <- confusion.matrix[2, 1]
  fpr_values_2[j] <- FP / (FP + TN)
  fnr_values_2[j] <- FN / (FN + TP)
  
  acc_2[j] = sum(diag(confusion.matrix))/sum(confusion.matrix)
  err_2[j] = 1 - sum(diag(confusion.matrix))/sum(confusion.matrix)
}

acc_knn = mean(acc_2); acc_knn # accuracy rate
err_knn = mean(err_2); err_knn # error rate
fpr_values_knn = mean(fpr_values_2); fpr_values_knn # FPR rate
fnr_values_knn = mean(fnr_values_2); fnr_values_knn # FPR rate

library(ROCR)
library(class)
has_diabetes <- knn(train = train_knn[, 2:14], test = diabetes[,2:14], cl = train_knn$Diabetes_binary, k = 88)
has_diabetes = as.numeric(has_diabetes)
pred_knn = prediction(has_diabetes, Diabetes_binary)
roc_knn = performance(pred_knn, measure="tpr", x.measure="fpr")
plot(roc_knn) 
auc_knn = performance(pred_knn , measure ="auc")
auc_knn@y.values[[1]] # AUC value


### LOGISTIC REGRESSION CLASSIFIER

acc_3 = numeric(n_folds)
err_3 = numeric(n_folds)
fpr_values_3 = numeric(n_folds)
fnr_values_3 = numeric(n_folds)


library(rpart)
library("rpart.plot")
for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train = rbind(train_nej, train_pos)
  test = rbind(test_nej, test_pos)
  
  model_logistic_regression<- glm(Diabetes_binary ~ .,
                                  data =train,family = binomial(link ="logit"))
  pred = predict(model_logistic_regression, newdata = test[,2:14], type = 'response')
  pred <- ifelse(pred > 0.5, 1, 0)
  confusion.matrix = table(pred, test[,1])
  
  TP <- confusion.matrix[2, 2]
  TN <- confusion.matrix[1, 1]
  FP <- confusion.matrix[1, 2]
  FN <- confusion.matrix[2, 1]
  fpr_values_3[j] <- FP / (FP + TN)
  fnr_values_3[j] <- FN / (FN + TP)
  
  acc_3[j] = sum(diag(confusion.matrix))/sum(confusion.matrix)
  err_3[j] = 1 - sum(diag(confusion.matrix))/sum(confusion.matrix)
  
  
}
acc_lr = mean(acc_3); acc_lr # accuracy rate
err_lr = mean(err_3); err_lr# error rate
fpr_values_lr = mean(fpr_values_3); fpr_values_lr # FPR rate
fnr_values_lr = mean(fnr_values_3); fnr_values_lr # FPR rate

library(ROCR)
has_diabetes = predict(model_logistic_regression, diabetes[,2:14], type ="response")

pred_lr = prediction(has_diabetes, Diabetes_binary)
roc_lr = performance(pred_lr, measure="tpr", x.measure="fpr")
plot(roc_lr) 
auc_lr = performance(pred_lr , measure ="auc")
auc_lr@y.values[[1]] # AUC value

### NAIVES BAYERS CLASSIFIER

acc_4 = numeric(n_folds)
err_4 = numeric(n_folds)
fpr_values_4 = numeric(n_folds)
fnr_values_4 = numeric(n_folds)

library(rpart)
for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train = rbind(train_nej, train_pos)
  test = rbind(test_nej, test_pos)
  
  model_nb<- glm(Diabetes_binary ~ .,
                 data =train,family = binomial(link ="logit"))
  pred = predict(model_logistic_regression, newdata = test[,2:14], type = 'response')
  pred <- ifelse(pred > 0.5, 1, 0)
  confusion.matrix = table(pred, test[,1])
  
  TP <- confusion.matrix[2, 2]
  TN <- confusion.matrix[1, 1]
  FP <- confusion.matrix[1, 2]
  FN <- confusion.matrix[2, 1]
  fpr_values_4[j] <- FP / (FP + TN)
  fnr_values_4[j] <- FN / (FN + TP)
  
  acc_4[j] = sum(diag(confusion.matrix))/sum(confusion.matrix)
  err_4[j] = 1 - sum(diag(confusion.matrix))/sum(confusion.matrix)
  
  
}
acc_nb = mean(acc_4); acc_nb # accuracy rate
err_nb = mean(err_4); err_nb # error rate
fpr_values_nb = mean(fpr_values_4); fpr_values_nb # FPR rate
fnr_values_nb = mean(fnr_values_4); fnr_values_nb # FPR rate

library(ROCR)
has_diabetes = predict(model_nb, diabetes[,2:14], type ="response")

pred_nb = prediction(has_diabetes, Diabetes_binary)
roc_nb = performance(pred_nb, measure="tpr", x.measure="fpr")
plot(roc_nb) 
auc_nb = performance(pred_dt , measure ="auc")
auc_nb@y.values[[1]] # AUC value

# COMPARING THE MODELS
# create vector containing all factors to examin goodness of fit
ACCURACY <- c(acc_decision_tree, acc_knn, acc_lr, acc_nb);ACCURACY
ERROR <- c(err_decision_tree, err_knn, err_lr, err_nb);ERROR
FPR <- c(fpr_values_decision_tree, fpr_values_knn, fpr_values_lr, fpr_values_nb);FPR
FNR <- c(fnr_values_decision_tree, fnr_values_knn, fnr_values_lr, fnr_values_nb);FNR
AUC <- c(auc_dc@y.values[[1]], auc_knn@y.values[[1]], auc_lr@y.values[[1]], auc_nb@y.values[[1]]);AUC

highest_accuracy <- which.max(ACCURACY); highest_accuracy
lowest_error <- which.min(ERROR); lowest_error
lowest_fpr <- which.min(FPR); lowest_fpr
lowest_fnr <- which.min(FNR); lowest_fnr
highest_auc <- which.max(AUC); highest_auc

# Therefore it can be concluded that the logistic regression model is the best model

# Comparing predicted response of logistic regression model and thebactual response variable 
response_best_model <- predict(model_logistic_regression, newdata = diabetes, type = 'response')

# Create side-by-side boxplots
boxplot(Diabetes_binary, response_best_model, 
        col = c("blue", "red"),
        xlab = "Response",
        main = "Actual vs Predicted Response")


