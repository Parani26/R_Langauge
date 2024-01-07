
##############  INTRODUCTION TO R
# use command + enter to run the code without copying and pasting

# when u get data
#attach,dim,head
#as factor relevant columns

# if id dont work dont panic and try churn$id 


# VECTOR CREATION
number<- c(2,4,6,8,10); number # creating a vector of numbers
inshalla <- c(9,3,5,7,1)
string<- c("weight", "height", "gender"); string # creating a vector of strings/characters
logic<- c(T, T, F, F, T); logic # creating a Boolean vector (T/F)
length(logic) # length of the vector

# COMMON COMMANDS
max(number) # maximum value of vector
min(number) # minimum value of vector
sum(number) # total of all the values in x
mean(number) # arithmetic average values in x
range(number) # min(x) and max(x)
cor(number, inshalla)# correlation bw vectors x and y # only for quantitative variables
sort(inshalla) # sorted version of x

# FUNCTION numeric()
number.2<- numeric(3); number.2 # creating a vector of zeros

# APPENDING/ADDING TWO VECTORS
new.number<- c(number, number.2); new.number
new.number1<- append(number, number.2);new.number1

# FUNCTION rep()
# rep(a,b): replicate the item a by b times where a could be a number or a vector
number.3<- rep(2,3); number.3
number.3<- rep(c(1,2),3); number.3 # need to have the c
new.string<- rep(string,2); new.string

# FUNCTION seq()
sequence1 <- seq(from=2, to=10, by=2);sequence1 # vector of 2,4,6,8,10
sequence2 <- seq(2,10,2);sequence2 # preferred # vector of 2,4,6,8,10
sequence3 <- seq(from=2, to=10, length = 5);sequence3 <- # vector of 2,4,6,8,10
sequence4 <- seq(10);sequence4 # a sequence from 1 up to 10, distance by 1 (automatic)

# FUNCTION matrix()
v <- c(1:6); v # creating vector of 1 to 6
m <- matrix(v, nrow=2, ncol=3); m # creates a matrix arranging data into the matrix created
m <- matrix(v, nrow=2, ncol=3, byrow=T); m # to fill the matrix by rows
m <- matrix(v, nrow=2, ncol=3, byrow=F); m # to fill the matrix by columns #automatic if nth specified

# just take note rows are horizontal and columns are vertical
a <- c(1,2,3,4)
b <- c(5,6,7,8)

# FUNCTION rbind()
ab_row <- rbind(a,b); ab_row

# FUNCTION cbind()
ab_col <- cbind(ab_row, c(9,10)); ab_col

# LIST IN R
#list will print out the data in 1D (vertically)
list.1 <- list(10.5, 20, TRUE, "Daisy"); list.1
x = c(2,4,6,8) # length 4
y = c(T, F, T) # length 3
list.2 = list(name1 = x, name2 = y); list.2 # assign names to list members
# referencing in list
list.2[1] # reference by index, indexing starts from 1 in R
list.2$name1 # reference by name


############## DATAFRAME IN R
# reading data frame
data1<-read.csv("crab.txt", sep = "", header = FALSE) 
data = data1[,-(1:3)] # REMOVE THE FIRST THREE COLUMNS (useless for analysis)

# number of rows and columns
nrow(data1) # number of rows
ncol(data1) # number of columns
nrow(data)
ncol(data)
dim(data)

# Reading data
# data \ seperated
dat = read.table("Colleges.txt",header =TRUE,sep= "\t") # read.table
# data , seperated
data1<-read.csv("crab.txt",sep = "", header = TRUE) # header True reads the first line of data as headers # read.csv
data1<-read.csv("crab.txt", sep = "", header = FALSE) # header false reads the headers as data

# to rename variables
varnames <- c("Subject", "Gender", "CA1", "CA2", "HW") # first way to rename
data2<-read.table("ex_1.txt", header = FALSE, col.names = varnames); data2
data4<-read.table("ex_1.txt", header = FALSE); data4
names(data4)[1:5] = c("Subject", "Gender", "CA1", "CA2", "HW") # second way to rename
#to rename column name 
names(hab)[4] = "status"

# reading specific data
attach(data1) # then dont need to read each variable
data1[1:8,] # first 8 rows 
data1[(nrow(data1) - 7):nrow(data1),] # last 8 rows
data1[,1:3] # first 3 column
data1[,(ncol(data1) - 2):ncol(data1)] # last 3 columns
data1[,1] # first column
data1[3,3] # value at 3rd row & 3rd column
data1[3,4] # value at 3rd row & 4th column
names(data1) # names of columns 
head(data1) # header and first 6 data sets

bankdata = read.csv("bank-sample.csv", header=TRUE)
head(bankdata[,c(9,16,17)]) # columns 9, 16, 17
table(bankdata$job) # To get summaries

# Selecting rows of data of specific data
attach(data3)
data3[Gender == "M",] # all the rows (observations) whose gender = M:
data3[Gender == "M" & CA2 > 85,] # all the rows (observations) whose gender = M and CA2>85

# add all values in the table
sum(table(data3$Gender))

# Converting categorical data to numerical
sur = ifelse(Survived == 'Yes', 1, 0) 
sur = as.factor(sur) 
titanic$sur = sur ; head(titanic) # changing data itself
# Be careful about using dot if your changing a variable

# drop a few columns to simplify the tree
drops<-c("age", "balance", "day", "campaign", 
         "pdays", "previous", "month", "duration")
banktrain <- banktrain [,!(names(banktrain) %in% drops)]
head(banktrain)

# SCALING THE INPUT FEATURES
standardized.X= scale(caravan[,-86]) # scaling all the dataset, except the last column, column 86 = RESPONSE, used to compare values of different scales like income and age
# if u create a model based on standardised values make sure to standarise the test case as well before running model on it
new = data.frame(X1 = 83, X2 = 57, X3= 2, X4 = 3)
standard = scale(rbind(data[,2:5], new) ) 
# dont standardise response varibales only input variables

#  WHILE LOOP
x = 1
while(x<=3) {print("x is less than 4")
             x = x+1}

x<-0; S<-0 # Find the sum of first 10 integers:
while(x<=10) {S<- S+ x
              x<-x+1}; S

#  FOR LOOP
S<-0 # Example: find the sum of first 10 integers
for(i in 1:10){S <-S+i}; S

x = c(2, 4, 3, 8, 10) # Find the mean of vector x
l = length(x) 
S = 0
for (i in 1:l){S = S + x[i]}
ave = S/l; ave

x = c(1:100) #Find the sum of all even numbers from 1 up to 100.
S = 0
for (i in 1:length(x)){
  if(x[i]%%2 ==0){S = S + x[i]} else {S = S}
}
print(S)

# defining a function
F1 <- function(down_payment, saved, monthly_return, portion_saved, salary, months){
  while (saved < down_payment){
    months <- months + 1
    saved <- saved + saved * monthly_return + salary * portion_saved
  }
  print(months)
}

# CONDITIONS WITH if()... else if()... else()
x = c(1:10); # a vector of numbers from 1 to 10
# we want to divide this vector into 3 subsets: 
# a set of all small numbers from 1 to 3
# a set of all medium numbers, from 4 to 7
# a set of large numbers from 8 to 10
S = numeric(0)
M = numeric(0)
L = numeric(0) 
for (i in 1:length(x)){
  if (x[i] <=3){S = append(S, x[i])} else if (x[i]< 8)
  {M = append(M, x[i])} else {L = append(L, x[i])}
}
print(S)
print(M)
print(L)

# FUNCTION ifelse()
x = c(1:8);x
y = ifelse(x%%2 == 0, "even", "odd"); y # if true first statement else second statement

#  REPEAT LOOP
i <-1 # EXAMPLE: print the first five integers
repeat {
  print (i)
  if(i ==5) { break } # break stops the code # must have
  i <- i+1
 }

S = 0 # Example: obtain the sum of first 5 integers
i <-1
repeat {
 S <-S+i;
 if(i ==5) { break }
 i <- i+1
 }; S

##############  BASIC PROBABILITY & STATISTICS
sales <- read.csv("yearly_sales.csv")
head(sales)
total = sales$sales_total # naming a specific column / variable
attach(sales)

n = length(total); n # number of observations in that variable
summary(total) # info on min, 1st quart, median, mean, 3rd quart and mx

# More information
range(total)
var(total)
sd(total)
IQR(total)
total[order(total)[1:5]] # The 5 smallest observations
total[order(total)[(n-4):n]] #The 5 largest observations

# HISTOGRAM in FREQUENCY
hist(total, freq=FALSE, main = paste("Histogram of total sales"),
     xlab = "total", ylab="Probability", 
     col = "blue")
lines(density(total), col = "red") # this is the density curve of "total"

# HISTOGRAM WITH DENSITY LINE
hist(total, freq=FALSE, main = paste("Histogram of total sales"),
     xlab = "total", ylab="Probability", 
     col = "blue", ylim = c(0, 0.0045)) # ylim determines the minimum and maximum values that will be displayed on the y-axis of the plot
lines(density(total), col = "red") # this is the density curve of "total"
# xlim = c(0, 3000)) , can add after col = "blue" to limit the x axis


# HISTOGRAM WITH NORMAL DENSITY 
hist(total, freq=FALSE, main = paste("Histogram of Total Sales"),
     xlab = "total sales", ylab="Probability", 
     col = "grey", ylim = c(0, 0.002)) # remove the lim if necssary
y <- dnorm(x, mean(total), sd(total))
x <- seq(0, max(total), length.out=length(sales_total))
lines(x, y, col = "red") # this is the normal density curve

hist(sales_total,prob = TRUE)
x= seq(50,90, length.out = length(sales_total))
y = dnorm(x, mean(sales_total), sd(sales_total))
lines(x,y, col = "red")

# The histogram shows that the sample is unimodal. Compared to the overlaid normal density curve,
# the distribution looks slightly right-skewed. Most of the observations are within a range of 0.5 to 6 
# – there are no observations that are separated from the rest. However, this does not mean there are no outliers.


fev <- read.csv("FEV.csv")
sex = fev$Sex
FEV = fev$FEV
# saving the values of a variable of categories of another variable e.g fev value of male and female
# plot seperate catgories
female = FEV[which(sex==0)] # or FEV[Sex==0]
male = FEV[which(sex==1)] # or FEV[Sex==1]

# comparing histogram models ***
# The shapes of the two histograms are quite different. 
# Both are unimodal, but for females, it is almost symmetrical but for males it is quite right-skewed.
# The median FEV for females is much lower than that for males (2.49 compared to 2.605).
# In addition, the variability in the male group is higher than the variability in the female group. 
# The respective IQR are 1.54 and 1.05.

# Split plot into 2, for spliting into 4 do (2,2)
opar <- par(mfrow=c(1,2)) #arrange a figure which has 1 row and 2 columns (to contain the 2 histograms)
#(2,2) for 4 histograms

hist(female, col = 2, freq= FALSE, main = "Histogram of Female FEV", ylim = c(0,0.52))
hist(male, col = 4, freq= FALSE, main = "Histogram of Male FEV", ylim = c(0,0.52))
par(opar) # Reset back the plot 
opar <- par(mfrow=c(1,1)) # last resort

#numerical summaries on data sets with specific observations
median(female)
IQR(female)
summary(female)
var(female)

# BOX PLOTS
boxplot(total, xlab = "Total Sales", col = "blue")
boxplot(FEV, col = 10, ylab = "FEV", main = "Boxplot of FEV")
outlier = boxplot(total)$out;outlier # get outlier values
length(outlier) #number of outliers

# There are 9 outliers. Check the information of these 9 outliers, ***
# we would see that all these outliers are males, most (8/9) are non-smokers, and they are rather tall.

#get the indexes of lutlier points
index = which(total %in% outlier)
index

#info on all outlier
sales[c(index),]

# QQ plot # to check if normally distrubuted
qqnorm(total, main = "QQ Plot", pch = 20)
qqline(total, col = "red")
 
# switch x and y
qqnorm(total, datax = TRUE, main = "QQ Plot",pch = 20)
qqline(total, datax = TRUE, col = "red")
# datax arguement optional

# From the qq plot, on the left tail, the sample quantiles are larger than expected (theoretical quantile) 
# hence the left tail is shorter than normal.
# On the right side, the sample quantiles are larger than expected, hence the right tail is longer than normal.
# Conclude: Combining with the histogram of FEV, it’s clear that the sample of FEV is not normally dis- tributed and quite right skewed.

# CORRELATION COEFFICIENT
order = sales$num_of_orders
cor(total, order) #0.75

# The computed correlation is quite high, and it is clear from the plot 
# that there is a strong positive linear association between the two variables overall.
# The range of FEV for males appears larger than the range for females, as does the range of heights.
# The variability of FEV at lower heights does seem to be slightly less than the variability of FEV 
# at greater heights.

# SCATTER PLOT
plot(order,total,pch=20,col="darkblue")
 
# BOX PLOTS OF MULTIPLE GROUPs
boxplot(total ~ sales$gender, col = "blue")
#assoc bw gender and sales


# 3 VARIABLES = SCATTER PLOT ADDING LEGEND
# plotting different observations of data
order = sales$num_of_orders
attach(sales)
# plotting data points separately according to categories of a variable
# x first, y second
plot(order,total, type = "n") # a scatter plot with no point added
points(order[gender=="M"],total[gender=="M"],pch = 2, col = "blue") # MALE
points(order[gender=="F"],total[gender=="F"],pch = 20, col = "red") # FEMALE
legend(1,7500,legend=c("Female", "Male"),col=c("red", "blue"), pch=c(20,2))
# (x = 1, y =7500) tells R the place where you want to put the legend box in the plot
# do note on the size of the points since the points added latter will overlay on the points added earlier
# hence, the points added latter should be chosen with smaller size so that they will not cover the points earlier


# scatter plot
x = c(0.4, 0.4, 0)
y = c(1, 0.4, 0.4)
plot(x,y, type = "n", xlab = "FPR", ylab = "TPR", ylim = c(0,1), xlim = c(0,1))
points(0.4,1, pch = 10, col = "red") # sigma = 0.3
points(0.4,0.4, pch = 10, col = "blue") # sigma = 0.6
points(0,0.4, pch = 10, col = "black") # sigma = 0.8
legend(0.6, 0.3, legend = c("sigma = 0.3", "sigma = 0.6", "sigma = 0.8"), 
       col = c("red", "blue","black"), pch = c(10, 10, 10))

# BARPLOT FOR CATEGORICAL VARIABLE
count = table(gender); count # frequency table
barplot(count)

# PIE CHART
count = table(gender); count # frequency table
pie(count)

# CATEGORIZING "ORDER"
order = sales$num_of_orders
# changing the data itself to small and large based on whether more or less than 5
sales$num_of_orders = ifelse(sales$num_of_orders <= 5, "small", "large") #replacing data with small or large based on condition
table(order)

# CONTINGENCY TABLE
table = table(gender,order);table
tab = prop.table(table, "gender");tab # proportion by gender, female / male total prob = 1
tab[1]/(1-tab[1]) # the odds of large order among FEMALES
tab[1]/tab[3] # same
tab[2]/(1-tab[2]) # the odds of large order among MALES
tab[2]/tab[4] # same
OR = (tab[1]/(1-tab[1]))/(tab[2]/(1-tab[2])); OR # 0.76
OR = (tab[1]/(tab[3]))/(tab[2]/(tab[4])); OR # same thing
# it means: the odds of larger orders among females is 0.76 times the odds of large orders among males.

############## RANDOMLY CHOOSING 2000 OBSERVATIOSN TO FORM TEST AND TRAIN X AND Y

# Be careful of 800 vs 200
n = dim(data1)[1] # sample size = 5822
test = sample(1:n, 2000) # sample a random set of 2000 indexes, from 1:n.
# response already removed from standardized.X
train.X=standardized.X[-test ,] #training set
test.X =standardized.X[test ,]  # test set
train.Y=Purchase[-test] # response for training set
test.Y =Purchase[test] # response for test set


# Or #

# Be careful of 800 vs 200
credit[ ,2:5] = lapply(credit[,2:5], scale)
train = sample (1:1000 , 800); #randomly sample a set of 800 indexes in 1:1000
train.data = credit[train,] # 800 data points for the train set
test.data = credit[-train,] # 200 data points for the test set
train.x = train.data[ ,2:5]
test.x = test.data[ ,2:5]
train.y = train.data[ ,1]
test.y = test.data[ ,1]


############## SPLITTING DATA INTO TRAIN AND TEST SETS so that equal number of observations
# test data : train data <- 1:4
# test and train data will have equal percentage of positive and negative response
n_folds = 5
folds_negative <- sample(rep(1:n_folds, length.out = dim(data4)[1]/2 )) 
# test data : train data <- 1:4
# test and train data will have equal percentage of positive and negative response
n_folds = 5
folds_negative <- sample(rep(1:n_folds, length.out = dim(diabetes)[1]/2 )) 
folds_positive <- sample(rep(1:n_folds, length.out = dim(diabetes)[1]/2 )) 
folds_positive <- sample(rep(1:n_folds, length.out = dim(diabetes)[1]/2 )) 

table(folds_negative)
table(folds_positive)

negative_data = diabetes[1:35346,] # Only negative response data
positive_data = diabetes[35347:70692,] # Only positive response data

acc_2 = numeric(n_folds)
err_2 = numeric(n_folds)
fpr_values_2 = numeric(n_folds)
fnr_values_2 = numeric(n_folds)

# then doing n fold
for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train_knn = rbind(train_nej, train_pos)
  test_knn = rbind(test_nej, test_pos)
  
  # use thsee new train and test to run the model
  # use acc_2,err_2,fpr_values_2,fnr_values_2 to add values in or other performance indicators when neccesary
}

table(folds_negative)
table(folds_positive)

negative_data = diabetes[1:35346,] # Only negative response data
positive_data = diabetes[35347:70692,] # Only positive response data

acc_2 = numeric(n_folds)
err_2 = numeric(n_folds)
fpr_values_2 = numeric(n_folds)
fnr_values_2 = numeric(n_folds)

# then doing n fold
for (j in 1:n_folds){
  test_nej <- which(folds_negative == j)
  test_pos <- which(folds_positive == j)
  
  train_nej = negative_data[ -test_nej, ]
  train_pos = positive_data[ -test_pos, ]
  test_nej = negative_data[test_nej, ]
  test_pos = positive_data[test_pos, ]
  
  train_knn = rbind(train_nej, train_pos)
  test_knn = rbind(test_nej, test_pos)
  
  # use thsee new train and test to run the model
  # use acc_2,err_2,fpr_values_2,fnr_values_2 to add values in or other performance indicators when neccesary
}

############## LINEAR REGRESSION 
#own function in r to derive eqn of model
simple <- function(x , y) {
  beta_1 <- (sum(x*y)- mean (y)* sum (x ))/( sum(x^2)- mean(x)* sum(x));
  beta_0 <- mean(y)- beta_1* mean(x) ;
  return(c( beta_0 , beta_1)) ;
}

simple(x,y) # manual way of calculating beta 0 and beta 1

# MODELLING
x = c( -1, 3, 5)
y = c( -1, 3.5 , 3)
M1 = lm(y~x) # shows beta 0 and beta 1
M1$fitted # y values when x fitted into eqn
# if you fit linear model for probability might get neagtive values: limitation

# also can do
M1 = lm(weight~width+spine,data = data)

# Is the fitted model significant?
# The fitted model, M, has F-test for the overall significance of the model 
# with extremely small p-value. Hence, model M is signicant.



# solve to calulate inverse
# t(x) is to tranpose the matrixz
# The %*% operator is used for matrix multiplication or dot product between two matrices

matrix <- function(x, y) {
  beta <- solve(t(x )%*% x )%*% t(x )%*% y
  return( beta )
} # to call use matrix(x,y)

# PREDICTING
new = data.frame(life_expectancy = 83, mortality = 57, infant = 2, alcohol = 3 ) # create dataframe of new point
predict(M1, newdata = new) 

# MODEL FOR HDB RESALE FLATS
resale = read.csv("hdbresale_reg.csv")
price = resale$resale_price
area = resale$floor_area_sqm
lm(price~area)$coef # coefficients of the model

# RSE
#Calculating RSE:
sqrt(sum((y - M1$fitted)^2)/(length(y) - 2))
summary(M) # Take the Residual standard error

# R^2
TSS = var(y)*(length(y) -1) # or
TSS = sum((y- mean (y)) ^2)
RSS =sum((y- M1$fitted )^2)
R2 = 1 - RSS/TSS; R2

# get r squared value
# multiple R-squared is the R square value 
summary(M1)$r.squared

# MULTIPLE LINEAR MODEL
set.seed(250) # randomises consistently 
x1 = rnorm(100) # generating values from normal disturbution
x2 = rnorm(100) 
y = 1 + 2*x1 -5*x2+ rnorm(100)
lm(y~x1+x2)

# How to present fitted model
# y = −15257.5 + 77.99x + 30569.1I(NW = 1).

#instal.packages("rgl")
library(rgl)
M.2 = lm(y~x1+x2)
# 3D plot to illustrate the data points
plot3d (x1 , x2 , y, xlab = "x1", ylab = "x2", zlab = "y",
       type = "s", size = 1.5 , col = "red")

coefs = coef(M.2) # get the coefficients of the model
a <- coefs[2]; a # coef of x1
b <- coefs[3]; b # coef of x2
c <- -1       # coef of y in the equation: ax1 + bx2 -y + d = 0.
d <- coefs[1] # intercept
planes3d (a, b, c, d, alpha = 0.5) # the plane is added to the plot3d above.


# MLR for HDB RESALE FLATS
resale = read.csv("hdbresale_reg.csv")
years_left = 2022 - resale$lease_commence_date # working a function on the column of data
price = resale$resale_price
area = resale$floor_area_sqm
M1 = lm(price ~ area)
summary(M1)
M2 = lm(price ~ area + years_left)
summary(M2) # on top p value is to see if that variable is significant or not (see number of stars) below p value is to see if overall model is significant

############## KNN CLASSIFIER

# Answering predict based on k value questions ***
# What is our prediction with K = 1? Why?
# Answer: Green. When K = 1, the one nearest point is Green (5th point, with distance of 1.41).
# Hence, for the test point, we classify it to the category that is the same as the category of the nearest point.

# What is our prediction with K = 3? Why?
# Answer: When K = 3, the three nearest points are the 5th (Green), 6th (Red) and 2nd (Red). 
# Hence, the test point will be classified as Red.

# K small or large?
# If the Bayes decision boundary (the gold standard decision boundary) in this problem is highly non-linear,
# then would we expect the best value for K to be large or small? Why?
# Answer: A small value for K, since it translates to a more flexible classification method.



# STOCK MARKET EXAMPLE
market = read.csv("Smarket.csv")
summary(market[,2:10]) #summary of data excluding response

#  PREPARING DATA TO FORM MODEL AND TO TEST MODEL
# to separate the data above to two parts: 
# one part is used to train the model
# another part is to test the model.
# We'll select the rows that belong to the years before 2005 to train model 
# index of the rows before year 2005 is in the vector "train":

train =(market$Year < 2005) # index of the rows in the years from 2001 to 2004 for training set

train.data = market[train,]
test.data  = market[!train,] # take the year 2005 as test set, the rest of rows in "market" is for testing model:

dim(train.data)
dim(test.data)

library(class)

# form a SET OF FEATURES for the training; and for testing: # choosing those columns and creating new data set
train.x = train.data[,c("Lag1","Lag2","Lag3","Lag4","Lag5")]
test.x = test.data[,c("Lag1","Lag2","Lag3","Lag4","Lag5")]
# form the RESPONSE for the training; and for testing:
train.y = train.data[,c("Direction")]
test.y = test.data[,c("Direction")]

#  FORMING MODEL = FORMING THE CLASSIFIER
set.seed(1)
knn.pred = knn(train.x,test.x,train.y,k=1)  # KNN with k = 1 # model comes up with fitted values of y for test.x # can vary k
# THE CLASSIFER FORMED BY KNN WAS CREATED, named knn.pred

# TO CHECK HOW GOOD THE CLASSIFER ABOVE IS, WE MAY USE ACCURACY:
confusion.matrix=table(knn.pred, test.y); confusion.matrix # does a comparison of actual and fitted
sum(diag(confusion.matrix))/sum(confusion.matrix) # 0.515873 # To calculate accuracy but not correct always make sure the down and up/ tp tn fp fn is in this order
# (55+75)/252 ~ 51.59% of the observations are correctly predicted

# to calculate precision of the classifier
set.seed (5)
knn.pred = knn(train.X,test.X,train.Y,k=1) # KNN with k = 1
confusion.matrix=table(test.Y,knn.pred)
confusion.matrix # Yes is in the second column/row
precision = confusion.matrix[2,2]/sum(confusion.matrix[,2])
precision

# N-FOLD CROSS VALIDATION 

#  A small example on dividing whole data set into n folds  #################
n_folds=3
Number_of_datapoints=12 # sample size
index=rep(1:n_folds,length.out = Number_of_datapoints);index # assigns 1 2 3 12 times 
s = sample(index); s # sample randomises 1 2 3 
# sample randomises the order
table(s) 
# dataset of 12 points is devided into 3 folds randomly, each fold has 4 points.
# the 4 points for each of 3 folds are selected from the dataset following s. For example,
# s = 3 1 1 3 2 2 2 2 1 3 1 3
# then, the first data point belongs to 3rd fold. The next 2 points belong to 1st fold, etc.


# 5-fold Cross-Validation for KNN with k=1, 5, 10, etc. for the data set Smarket.csv
X=market[,c("Lag1","Lag2","Lag3","Lag4","Lag5")] # columns of explanatory features
Y=market[,c("Direction")] # response
dim(market) # 1250 data points/observations

n_folds=20
folds_j <- sample(rep(1:n_folds, length.out = dim(market)[1] )) 
table(folds_j)

err=numeric(n_folds) # create empty vector of 0s that is the length of n folds 
acc=numeric(n_folds)

for (j in 1:n_folds) {
	test_j <- which(folds_j == j) # get the index of the points that will be in the test set
	pred <- knn(train=X[ -test_j, ], test=X[test_j, ], cl=Y[-test_j ], k=1) # KNN with k = 1, 5, 10, etc # pred is the fitted value
  # -test_j means all data not in testj
	# test_j means all data in testj
	
	err[j]=mean(Y[test_j] != pred) # err j is the mean error of all the data rows
	acc[j]=mean(Y[test_j] == pred) # acc j is the mean accuracy of all the data rows
      # this acc[j] = sum(diag(confusion.matrix))/sum(confusion.matrix), where confusion.matrix=table(Y[test_j],pred)
}
err
acc
error=mean(err); error # mean of all the folds' errors
accur=mean(acc); accur # mean of all the folds' erors

# doing knn for k from 1 to 100
K = 100 # can try KNN with k = 1,2,...K.
accuracy=numeric(K) # to store the average accuracy of each k.
acc=numeric(n_folds) # to store the accuracy for each iteration of n-fold CV

for (i in 1:K){
  for (j in 1:n_folds) {
    test_j <- which(folds_j == j) # get the index of the points that will be in the test set
    pred <- knn(train=X[ -test_j, ], test=X[test_j, ], cl=Y[-test_j ], k=i) # KNN with k = 1, 5, 10, etc
    
    acc[j]=mean(Y[test_j] == pred) 
    # this acc[j] = sum(diag(confusion.matrix))/sum(confusion.matrix), where confusion.matrix=table(Y[test_j],pred)
  }
  accuracy[i] = mean(acc)
}

# to find three biggest accuracy of k
max(accuracy)
sort(accuracy)[98:100] # the three largest accuracy
index = which(accuracy == max(accuracy)) ; index # give index which is also the value of k.
plot(x=1:100, accuracy, xlab = "K")
abline(v = index, col = "red", ) # drawing a vertical straight line at the point which in this case is max

# another way to calculate accuracy
accuracy = sum(diag(confusion.matrix))/sum(confusion.matrix); accuracy

# if doing nested loop takes too long do both loops separately
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


############## DECISION TREE
#install.packages("rpart")
#install.packages("rpart.plot")
library("rpart")
library("rpart.plot")
bankdata <- read.csv("bank-sample.csv")
fit <- rpart(subscribed ~job + marital + education+default + 
               housing + loan + contact+poutcome,
             method="class",
             data=bankdata,
             control=rpart.control(minsplit=1),
             parms=list(split='information')
             #or use "gini" for split
)
# which are the important features? ***
# It seems the sepal length and sepal width are not important in the classification 
# while the petal length and petal width are more important.


# there is another argument in rpart.control, that is cp.
# smaller values of cp correspond to decision trees of larger sizes, 
# and hence more complex decision surfaces.

# method = "anova", "poisson", "class" or "exp"
# If response is a survival object, then method = "exp" is assumed, 
# if response has 2 columns then method = "poisson" is assumed, 
# if response is a factor then method = "class" is assumed, 
# otherwise method = "anova" is assumed
# minslpit = 1: a stem is created when data have at least one observation in that stem
# split = 'information' or 'gini'

# To plot the fitted tree:
rpart.plot(fit, type=4, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)
rpart.plot(fit, type=4, extra=2)# can try with extra = 4 to see the difference #proportion instead of fraction
rpart.plot(fit, type=3, extra=2)# can try with extra = 4 to see the difference
rpart.plot(fit, type=3, extra=2, varlen=0, faclen=0, clip.right.labs=FALSE)
# only the bottom node will show
# for implementation specifics refer to notes

#varlen 4: variable name only 4 letters
#farlen 4: factor name only 4 letters
#varlen = length of variable's name,varlen = 0 means full name
#faclen = length of category's name, faclen = 0 means full name
#clip.right.labs: TRUE means: don't print the name of variable for the right stem
# You can try with varlen = 4 to see the difference compared to varlen = 0.
# name of variable only on left branch not right one if right.labs = TRUE

length(bankdata$poutcome)
table(bankdata$poutcome)

# ENTROPY PLOT 
p=seq(0,1,0.01)
D=-(p*log2(p)+(1-p)*log2(1-p))
plot(p,D,ylab="D", xlab="P(Y=1)", type="l")


# Calculating conditional entropy when 'poutcome' is splitted 
# as x1 = failure, other, unknown and x2 = success
x1=which(bankdata$poutcome!="success") # index of the rows where poutcome = x1
length(x1) # 1942 rows that the value of poutcome = x1.
x2=which(bankdata$poutcome=="success") # index of the rows where poutcome = x2
length(x2) # 58 rows that the value of poutcome = x2 = success
table(bankdata$subscribed[x1]) 
# counting how many "yes" and how many "no" for Subscribed among those with poutcome = x1
# among 1942 customers with poutcome = x1, 179 subscribed (179 yes), and 1763 no.
table(bankdata$subscribed[x2]) 
# counting how many "yes" and how many "no" for Subscribed among those with poutcome = x2
# among 58 customers with poutcome = x2, 32 subscribed (32 yes), and 26 no.

# Calculating conditional entropy when 'poutcome' is splitted 
# as x1 = success, other, unknown and x2 = failure
x1=which(bankdata$poutcome!="failure")
x2=which(bankdata$poutcome=="failure")
table(bankdata$subscribed[x1])
table(bankdata$subscribed[x2])

#  PLAYING GOLF EXAMPLE
library("rpart") # load libraries
library("rpart.plot")
play_decision <- read.table("DTdata.csv",header=TRUE,sep=",")
head(play_decision)

fit <- rpart(Play ~ Outlook + Temperature + Humidity + Wind,
             method="class",
             data=play_decision,
             control=rpart.control(minsplit=1),
             parms=list(split='information'))

rpart.plot(fit, type=4, extra=2)
newdata <- data.frame(Outlook="rainy", Temperature="mild",
                      Humidity="high", Wind=FALSE)
newdata
predict(fit,newdata=newdata,type="prob")
predict(fit,newdata=newdata,type="class")


# another way to predict with decision tree

predict(fit, newdata <- data.frame(Outlook="rainy", Temperature="mild",
                                   Humidity="high", Wind=FALSE), type = 'class') # get yes or no
predict(fit, newdata <- data.frame(Outlook="rainy", Temperature="mild",
                                   Humidity="high", Wind=FALSE), type = 'prob') # get probability

# If unsure
??rpart.plot

# Carrying 5 fold cv on data where each train and test set has equal obersvations of 3 diff type of flowers
library(rpart)
set.seed(555)
iris <- read.csv("iris.csv")
head(iris)
attach(iris)
table(species)

# N-Cross Validation for decision tree
# making sure equal number of the three responses in each fold and there fore eual in test and train

n_folds=5
folds_setosa <- sample(rep(1:n_folds, length.out = dim(iris)[1]/3 )) 
folds_virginica <- sample(rep(1:n_folds, length.out = dim(iris)[1]/3 )) 
folds_versicolor <- sample(rep(1:n_folds, length.out = dim(iris)[1]/3 )) 

table(folds_setosa)
table(folds_virginica)
table(folds_versicolor)

setosa_data = iris[1:50,]
virginica_data = iris[51:100,]
versicolor_data = iris[101:150,]

acc = numeric(n_folds)

for (j in 1:n_folds) {
  test_set <- which(folds_setosa == j)
  test_virg <- which(folds_virginica == j)
  test_vers <- which(folds_versicolor == j)
  
  train_setosa = setosa_data[ -test_set, ]
  train_virginica = virginica_data[ -test_virg, ]
  train_versicolor = versicolor_data[ -test_vers, ]
  
  test_setosa = setosa_data[test_set, ]
  test_virginica = virginica_data[test_virg, ]
  test_versicolor = versicolor_data[test_vers, ]
  
  train = rbind(train_setosa, train_virginica, train_versicolor)
  test = rbind(test_setosa, test_virginica, test_versicolor)
  
  fit.iris <- rpart(species ~ ., 
  #when using dot be careful that you dont include response inside if your "species" is a derived response variable
  #from the exact response variable
                    method = "class", data =train, control = rpart.control( minsplit =1),
                    parms = list( split ='gini'))
  
  pred = predict(fit.iris, newdata = test[,1:4], type = 'class')
  confusion.matrix = table(pred, test[,5])
  acc[j] = sum(diag(confusion.matrix))/sum(confusion.matrix)
  
}

acc
mean(acc)


#calculating best cp for decision tree
#tut 7 qn 3

library(rpart) 
library(rpart.plot)

banktrain <- read.csv("bank-sample.csv", header=TRUE)
dim(banktrain)

# total records in dataset
n=dim(banktrain)[1]; n

# drop a few columns to simplify the tree
drops<-c("age", "balance", "day", "campaign", 
         "pdays", "previous", "month", "duration")
banktrain <- banktrain [,!(names(banktrain) %in% drops)]
head(banktrain)

length(which(banktrain[,9] =="yes")) 
# indexes of all customers that equal that observtaion
# 211 out of 2000 customers subscribed.

# We'll randomly split data into 10 sets of (about) equal size
# regardless of percentage of "yes" in each set.

n_folds=10
folds_j <- sample(rep(1:n_folds, length.out = n))
# this is to randomly sample the indexes of subsets for the observation
# table(folds_j)

cp=10^(-5:5); length(cp) # take not of cp
misC=rep(0,length(cp)) # a vector to record the rate of mis-classification for each cp

# doing n cv for each cp
for(i in 1:length(cp)){
  misclass=0
  for (j in 1:n_folds) {
    test <- which(folds_j == j)
    train = banktrain[-c(test),]
    fit <- rpart(subscribed ~ job + marital + 
                   education+default + housing + 
                   loan + contact+poutcome, 
                 method="class", 
                 data=train,
                 control=rpart.control(cp=cp[i]),
                 parms=list(split='information'))
    
    new.data=data.frame(banktrain[test,c(1:8)]) # check if response there if not there remove it before inputting test or train data
    ##predict label for test data based on fitted tree
    pred=predict(fit,new.data,type='class')
    misclass = misclass + sum(pred!=banktrain[test,9])
  }
  misC[i]=misclass/n # total misclass for each n fold added and averaged
}

plot(-log(cp,base=10),misC,type='b') # plot misclassifictaion rate against cp

#'p': This is the default. It creates a scatterplot with points.
# 'l': It creates a line plot.
# 'b': It creates a plot with both points and lines connecting them.
# 'c': This is similar to 'b,' but without lines connecting the points.
# 'o': This is also similar to 'b,' but the points are overplotted on top of the lines.

## determine the best cp in terms of
## misclassification rate


best.cp =cp[which(misC == min(misC))] ; best.cp
# 0.01
# this is the value of cp that gives the lowest mis-classification rate

## Fit decision tree with that smallest cp
fit <- rpart(subscribed ~ job + marital + education+default + housing + loan + contact+poutcome, 
             method="class", 
             data=banktrain,######
             control=rpart.control(cp=best.cp),
             parms=list(split='information'))

# to get the tree plotted:
rpart.plot(fit, type=4, extra=2, clip.right.labs=FALSE, varlen=0)#, faclen=3)

# ROC for Decision Trees:
titanic <- read.csv("Titanic.csv")
attach(titanic)

sur = ifelse(Survived == 'Yes', 1, 0) 
sur = as.factor(sur) 
titanic$sur = sur ; head(titanic)

M2<- rpart(sur ~ Class + Sex + Age, 
           method ="class",
           data = titanic,
           control = rpart.control(minsplit = 1),
           parms = list(split ='information'))

pred.M2 = predict(M2, titanic[,1:3], type='class') # no need to just take yes cuz its already a vector 
pred.M2= as.numeric(paste(pred.M2)) # changing format to numeric as pred is in classes

library(ROCR)
pred_dt = prediction(pred.M2, titanic$sur)
roc_dt = performance(pred_dt, measure="tpr", x.measure="fpr")
plot(roc_dt, add = TRUE) # add = True to add graoh only if axis is same (range of values on axes are same)

legend("bottomright", c("Naive Bayes","Decision Trees"),col=c("red","black"), lty=1)


auc2 = performance(pred_dt , measure ="auc")
auc2@y.values[[1]] # 0.683162

# WHEN WE FORM THE TREE USING SURVIVED (YES/NO) AND GET THE PROBABILITIES
# INSTEAD OF GETTING THE CLASS FOR OUTCOME, THEN:

# ROC for decision tree

M2<- rpart(Survived ~ Class + Sex + Age, 
           method ="class",
           data = titanic,
           control = rpart.control(minsplit = 1),
           parms = list(split ='information'))


rpart.plot(M2 , type =4, extra =2, clip.right.labs = FALSE , varlen =0, faclen =0)


# by probabilities

pred.M2 = predict(M2, newdata = titanic[,1:3], type = 'prob') # will get probbalities yes and no
pred.M2 = predict(M2, titanic[,1:3], type='prob')
score2 = pred.M2[,2] # here you have to take out the probability for yes since its not a vector in the necessary format

#prediction function only takes in numerical
pred_dt = prediction(score2, titanic$Survived)
roc_dt = performance(pred_dt, measure="tpr", x.measure="fpr")
plot(roc_dt)

legend("bottomright", c("Naive Bayes","Decision Trees"),col=c("red","black"), lty=1)


auc2 = performance(pred_dt , measure ="auc")
auc2@y.values[[1]] 
# 0.7262628

############## NAIVES BAYERS CLASSIFIER

# EXAMPLE 1: CLASSIFYING FRUITS
fruit.dat= read.csv("fruit.csv")
#Long/Sweet/Yellow: 1 = Yes, 0 = No
fruit.dat<- data.frame(lapply(fruit.dat, as.factor))
# lapplY to as.factor column by column
# as.factor to declare numeric factor as a categorical factor; changing type of example
# can as.factor one by one 
# for example can fruit.dat[,1] <- as.factor(fruit.dat[,1]) can do this manually for each column

head(fruit.dat)
attach(fruit.dat)

table(Long)
table(Sweet)
table(Yellow)

#Install package 'e1071' first
#install.packages("e1071")
library(e1071)

model <- naiveBayes(Fruit ~ Long+Yellow+Sweet,fruit.dat)

newdata <- data.frame(Long=1,Sweet=1, Yellow=0) # if for example Class = 2nd must use "2nd" and not 2nd
newdata <- data.frame(lapply(newdata, as.factor)) # as.factor here to create it
results <- predict (model,newdata,"raw") # probability of each response
results
results <- predict (model,newdata,"class") # default setting # Yes or No
results

# EXAMPLE 2: EMPLOYEE & ONSITE EDUCALTIONAL PROGRAM
sample <- read.table("sample1.csv",header=TRUE,sep=",")
# Enrolls = RESPONSE with 2 categories

# PART 1: MANUAL FORMING NAIVE BAYES CLASSIFIER ######
traindata <- as.data.frame(sample[1:14,])
testdata <- as.data.frame(sample[15,]) # in this instance no response variable for this row in the first place so no need to remove response variable later

# get the probability of each categories of the response, Compute the probabilities
tprior <- table(traindata$Enrolls);tprior
tprior <- tprior/sum(tprior); tprior # gives data in terms of probability, quite intuitive

# Get P(X = xi|Y = yj): row-wise proportion for each feature
# in table() first one is y second one is x
ageCounts <- table(traindata[,c("Enrolls", "Age")]);ageCounts # Get P(X = xi|Y = yj): row-wise proportion for feature AGE
ageCounts <- ageCounts/rowSums(ageCounts); ageCounts
incomeCounts <- table(traindata[,c("Enrolls", "Income")]) # Get P(X = xi|Y = yj): row-wise proportion for feature INCOME
incomeCounts <- incomeCounts/rowSums(incomeCounts);incomeCounts
jsCounts <- table(traindata[,c("Enrolls", "JobSatisfaction")]) # Get P(X = xi|Y = yj): row-wise proportion for feature JOBSATISFACTION
jsCounts <- jsCounts/rowSums(jsCounts);jsCounts
desireCounts <- table(traindata[,c("Enrolls", "Desire")]) # Get P(X = xi|Y = yj): row-wise proportion for feature DESIRE
desireCounts <- desireCounts/rowSums(desireCounts);desireCounts

# Applying the formular
# Proportion that point 15 will be "Yes" for the outcome is proportional to:
prob_survived <-
  classCounts["Yes","2nd"]*
  genderCounts["Yes","Female"]*
  ageCounts["Yes","Adult"]*
  tprior["Yes"]

prob_not_survived <-
  classCounts["No","2nd"]*
  genderCounts["No","Female"]*
  ageCounts["No","Adult"]*
  tprior["No"]

prob_survived
prob_not_survived

# MAKING DECISION:
prob_survived/prob_not_survived #4.115226. Hence the 15th observation should be classified as YES.

# PART 2: USE PACKAGE e1071 FORMING NAIVE BAYES CLASSIFIER ######
install.packages("e1071s")
library(e1071)

# testdata  and train data already defined above but rewriting here for simplicity
traindata <- as.data.frame(sample[1:14,])
testdata <- as.data.frame(sample[15,]) # the response is not inside so no need to remove response variable

model <- naiveBayes(Enrolls ~ Age+Income+JobSatisfaction+Desire, traindata) #, laplace=0) # can have data = traindata as well
#laplace helps add a bias in case the probability of any calculation is 0 
results <- predict(model,testdata,"raw"); results
# raw - we want probabilities plotted and given to us
# class - we wont get detailed probability but final outcome
results[2]/results[1] # 4.115226

results <- predict(model,testdata,"class"); results # if you want yes or no

results <- predict(model,traindata[,1:4],"class"); results
cbind(results, traindata[,5] )
data.frame(results, traindata[,5]) 
############### BANK-SAMPLE DATA ==> ROC and AUC

banktrain <- read.csv("bank-sample.csv", header=TRUE)
dim(banktrain)
head(banktrain)

# drop a few columns to simplify the tree
drops<-c("age", "balance", "day", "campaign", 
         "pdays", "previous", "month", "duration")
banktrain <- banktrain [,!(names(banktrain) %in% drops)]
head(banktrain)

# TESTING DATA SET
banktest <- read.csv("bank-sample-test.csv")
banktest <- banktest[,!( names ( banktest ) %in% drops )]

library(e1071)

# build the naive Bayes classifier
nb_model <- naiveBayes( subscribed ~., data = banktrain)

# perform on the test set BUT we need to remove the response column first
head(banktest);
ncol(banktest) # number of columns = 11. Response = 11th column.

nb_prediction <- predict(nb_model, newdata = banktest[,-ncol(banktest)], type ='raw') # remove last column
# if you use class instead of raw easier to compare later
# this is the predicted response for the test set
nb_prediction
cbind(nb_prediction, banktest[,ncol(banktest)])


# PLOT ROC CURVE FOR THE NAIVE BAYES CLASSIFIER ABOVE:
#install.packages("ROCR") 
# https://cran.r-project.org/web/packages/ROCR/ROCR.pdf

library(ROCR)
score <- nb_prediction[, c("yes")] # score is the conditional prob from Naive Bayes classifier for each test point
actual_class <- banktest$subscribed == 'yes' # actual response is 0 or 1

pred <- prediction(score , actual_class) # this is to "format" the input so that we can use the function in ROCR to get TPR and FPR
# repackage
perf <- performance(pred , "tpr", "fpr")
# calac diff tor and fpr values based on diff treshhold value

plot (perf, lwd =2) # lwd is to specify how thick the curve is
abline (a=0, b=1, col ="blue", lty =3)


# COMPUTE AUC FOR NAIVE BAYES CLASSIFIER:
auc <- performance(pred , "auc")@y.values[[1]] #to unlist #auc <- unlist(slot (auc , "y.values"))
auc
# auc is used to compare between Naive Bayes methd with other 
# methods such as linear model, logistic model, DT, etc. 
# the one with larger auc value is better.

# VISUALIZE ON HOW THE THRESHOLD CHANGES WILL CHANGE TPR AND FPR:
threshold <- round (as.numeric(unlist(perf@alpha.values)) ,4) # convert to numeric form
fpr <- round(as.numeric(unlist(perf@x.values)) ,4) #round to four decimal places
tpr <- round(as.numeric(unlist(perf@y.values)) ,4)
#see what are the chosen treshold values
# storing for and tpr values in a vector

# adjust margins and plot TPR and FPR
par(mar = c(5 ,5 ,2 ,5))
# mar = a numerical vector of the form c(bottom, left, top, right) = c(5,4,4,2)
# http://127.0.0.1:14187/library/graphics/html/par.html

plot(threshold ,tpr , xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "blue")
par( new ="True") # treshhold vs tpr
plot(threshold ,fpr , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "red" )
axis(side =4) # to create an axis at the 4th side
# treshhold vs fpr
mtext(side =4, line =3, "False positive rate")
text(0.4 ,0.05 , "FPR")
text(0.6 ,0.35 , "TPR")

cbind(threshold,fpr,tpr)

#Manually calculating the probability of survivng or not
#classCounts, genderCounts, and ageCounts are conditional probbailities
#tpior is table(response)
prob_survived <-
  classCounts["Yes","2nd"]*
  genderCounts["Yes","Female"]*
  ageCounts["Yes","Adult"]*
  tprior["Yes"]


# CONCLUDE: DECISION TREE IS BETTER THAN NAIVE BAYES

############## LOGISTIC CURVE

z = seq ( -10 ,10 ,0.1);
logistic = function (z) {exp(z)/(1+ exp(z))}

plot(z, logistic(z), xlab ="x", ylab ="p", lty =1, type ='l')


#  DATA SET ON CUSTOMER CHURN

churn = read.csv("churn.csv")
head(churn)

churn$Churned = as.factor(churn$Churned)
churn$Married = as.factor(churn$Married)
churn= churn[,-1] #Remove ID column

attach(churn)

table(Churned)
prop.table(table(Churned))

# LOGISTIC REGRESSION
# LOGISTIC MODEL

sur = ifelse(Survived == 'Yes', 1, 0)# response must be of 0 and 1 to fit the model
sur = as.factor(sur)
titanic$sur = sur ; head(titanic)
# Remb to do if not already in ones and zeros


M1<- glm( Churned ~., data =churn,family = binomial)
# ~ means depending on 
# . replaces all columns
summary(M1)

# FEMALE IS REFERENCE. MALE IS INDICATED BY INDICATOR since male can be seen in summary
# coefficient is estimated = -2.4201. 
# It means, given the same condition on the class and age,
# when comparing to a female, the LOG-ODDS of survival for a male is less than by 2.42.
# It means, the ODDS of survival of a male passenger will be less than that of a female by
# e^2.42 = 11.25 TIMES.


M2<- glm( Churned ~ Age + Married + Churned_contacts,
          data =churn,family = binomial(link ="logit"))
#Link = logit selected by default
summary(M2)

M3<- glm( Churned ~Age + Churned_contacts,
          data =churn,family = binomial(link ="logit"))
summary(M3)

predict(M3, newdata = data.frame(Age = 50, Churned_contacts = 5), type = 'response')
# type = 'response' means we want to get the Pr(Y = 1).


# ROC CURVE FOR LOGISTIC MODEL

library(ROCR)
prob = predict(M3, type ="response")
# dont need to specify data (new data = ...) cuz R will automatically predict the full data when you dont spcify

# above is to predict probability Pr(Y = 1) for each point in the training data set, using M3
# type = c("link", "response", "terms"). 
# http://127.0.0.1:14187/library/stats/html/predict.glm.html

pred = prediction(prob , Churned )
roc = performance(pred , "tpr", "fpr") # extract tpr and fpr
auc = performance(pred , measure ="auc")
auc@y.values[[1]] # area under the curve
plot(roc , col = "red", main = paste(" Area under the curve :", round(auc@y.values[[1]] ,4))) # rounding auc value to 4 dec places


# ROC for Logistic Regresison:(Another way)
pred = predict(M2, type="response") # type = response to get the probability of survived
pred_log = prediction(pred, titanic$Survived)
roc_log = performance(pred_log, measure="tpr", x.measure="fpr")
plot(roc_log, col = "red")

auc1 = performance(pred_log , measure ="auc")
auc1@y.values[[1]] # 0.7597259



# HOW TPR, FPR CHANGE WHEN THRESHOLD CHANGES:

# extract the alpha(threshold), FPR , and TPR values from roc
# treshold is alpha is here
length(alpha) #328 couple values of fpr tpr used hereto create curve
fpr <- round(as.numeric(unlist(roc@x.values)) ,4)
tpr <- round(as.numeric(unlist(roc@y.values)) ,4)

x = cbind(alpha, tpr, fpr)
x

# adjust margins and plot TPR and FPR
par(mar = c(5 ,5 ,2 ,5))

plot(alpha ,tpr , xlab ="Threshold", xlim =c(0 ,1) ,
     ylab = "True positive rate ", type ="l", col = "blue")
par( new ="True")
plot(alpha ,fpr , xlab ="", ylab ="", axes =F, xlim =c(0 ,1) , type ="l", col = "red" )
axis( side =4) # to create an axis at the 4th side
mtext(side =4, line =3, "False positive rate")
text(0.18 ,0.18 , "FPR")
text(0.58 ,0.58 , "TPR")

# there are some metrics that can help to choose a threshold: G-mean; Youden’s J statistic; etc

############## K MEANS

# HDB FLATS GROUPING
hdb=read.csv("hdbresale_cluster.csv")
head(hdb)
dim(hdb)

table(hdb$flat_type)

set.seed(1)


plot(x=hdb$floor_area_sqm, y=hdb$amenities,
     xlab="Floor area in sqm", ylab="Number of amenities", col="red") # plot one variable against another

kout <- kmeans(hdb[,c("floor_area_sqm","amenities")],centers=2)

plot(hdb$floor_area_sqm, 
     hdb$amenities, 
     col=kout$cluster)


kout$cluster # A vector of integers (from 1:k) indicating the cluster to which each point is allocated.

kout$centers # A matrix of cluster centres.

kout$size # The number of points in each cluster.

kout$withinss # Vector of SS_k, one value per cluster

kout$tot.withinss # Total within-cluster sum of squares = WSS


# PLOT TO SEE HOW WSS CHANGES WHEN K CHANGES

K = 10 # WE'LL TRY WITH k = 1, ...10.

wss <- numeric(K)

for (k in 1:K) { 
  wss[k] <- sum(kmeans(hdb[,c("floor_area_sqm","amenities")],centers=k)$withinss )
}

plot(1:K, wss, col = "red", type="b", xlab="Number of Clusters",  ylab="Within Sum of Squares")



# GRADE GROUPING

set.seed(1)
grade = read.csv("grades_km_input.csv")
head(grade)

attach(grade)

# VISUALIZE DATA SET BY FEATURES:
plot(grade[,2:4]) # plot a matrix
# PROPOSE: MIGHT BE 3 OR 4 GROUPS


kout <- kmeans(grade[,c("English","Math","Science")],centers=3)



plot(English, Science, col=kout$cluster)
plot(English, Math, col=kout$cluster)
plot(Math, Science, col=kout$cluster)

kout$withinss

# PLOT WSS vs K TO PICK OPTIMAL K:

K = 15 
wss <- numeric(K)

for (k in 1:K) { 
  wss[k] <- sum(kmeans(grade[,c("English","Math","Science")], centers=k)$withinss)
}


plot(1:K, wss, col = "blue", type="b", xlab="Number of Clusters",  ylab="Within Sum of Squares")

# comments:

# WSS is greatly reduced when $k$ increases from 1 to 2. 
# Another substantial reduction in WSS occurs at $k = 3$.

# However, the improvement in WSS is fairly linear for $k > 3$.
# Therefore, the $k$-means analysis will be conducted for $k = 3$.

# The process of identifying the appropriate value of k is
# referred to as finding the ``elbow'' of the WSS curve

# Calculate manually the eucleadin distance
x1 = c(1, 1.5, 3, 3.5, 4.5)
x2 = c(1,2,4,5,5)

plot(x1, x2, pch = 20, col = "blue")

text(1.1,1.1,"A") # labelling points on plot
text(1.6, 2.2, 'B')
text(3.1, 4.1, 'C')
text(3.63, 5, 'D')
text(4.35, 5, 'E')

# Adding the starting centroids 
points(2,2, pch = 2, col = 'red') # adding points to plot
text(2.2, 2.1, 'C-P') # adding text to plot
points(4,4, pch = 10, col = 'darkgreen')
text(4,3.8, 'C-Q')

# Adding the new centroids after the first iteration:
points(1.25, 1.5, col = 'red', pch = 2)
text(1.35, 1.4, 'C-P-new')
points(11/3, 14/3, col = 'darkgreen', pch = 10)
text(11/3, 4.5, 'C-Q-new')

# woring the k function on created values
data = data.frame(x1, x2)
data
kout = kmeans(data, centers = 2)
kout$withinss
kout$tot.withinss

##############  ASSOCIATION RULES 


#install.packages('arules')
#install.packages('arulesViz')

# documentation of package 'arules'
# https://cran.r-project.org/web/packages/arules/arules.pdf

library('arules')
library('arulesViz')

data(Groceries)

?Groceries
# three parts itemsets, xxx ,datasets

summary(Groceries) # How does it look in the data set. # summary of all 3 parts

# this link below is helpful to understand this special data
# https://www.jdatalab.com/data_science_and_data_mining/2018/10/10/association-rule-transactions-class.html#:~:text=The%20Transactions%20Class,-The%20arules%20package&text=The%20Groceries%20data%20set%20contains,to%20read%20the%20Groceries%20data.

inspect(head(Groceries)) # the first 6 transactions # itemsets

Groceries@itemInfo[1:10,] # item
Groceries@data[,100:110] # data # row and column opposite

# the items for first 5 transactions:
apply(Groceries@data[,1:5], 2,
      function(r) paste(Groceries@itemInfo[r,"labels"], collapse=", ")) #function r is to convert dot and line to readable data

# the items for 100th to 105-th transactions:
apply(Groceries@data[,100:105], 2,
      function(r) paste(Groceries@itemInfo[r,"labels"], collapse=", "))

#  GETTING THE FREQUENT 1-ITEMSETS:

itemsets.1 <- apriori(Groceries, parameter=list(minlen=1, maxlen=1,
                                                support=0.02, target="frequent itemsets")) # if support of itemset more than 0.02 considered frequent

summary(itemsets.1)

# minlen = 1: frequent itemset has at least 1 item
# maxlen = 1: frequent itemset has max = 1 item
# set both 'minlen = 1' and 'maxlen = 1' means we want frequent itemset that has only 1 item.


# list the most 10 frequent 1-itemsets:
inspect(head(sort(itemsets.1, by = "support"), 10))

# list all the 59 frequent 1-itemsets:
inspect(sort(itemsets.1, by ="support"))


#  GETTING THE FREQUENT 2-ITEMSETS:

itemsets.2 <- apriori(Groceries, parameter=list(minlen=2, maxlen=2,
                                                support=0.02, target="frequent itemsets"))

summary(itemsets.2)

# list all the frequent 2-itemsets:
inspect(sort(itemsets.2, by ="support"))

# list of most 10 frequent 2-itemsets:
inspect(head(sort(itemsets.2, by = "support"), 10))


#  GETTING THE FREQUENT 3-ITEMSETS:


itemsets.3 <- apriori(Groceries, parameter=list(minlen=3, maxlen=3,
                                                support=0.02, target="frequent itemsets"))

summary(itemsets.3)
inspect(sort(itemsets.3, by ="support"))

# only TWO frequent itemsets that meets the minimum support of 0.02.

#  GETTING THE FREQUENT 3-ITEMSETS:

itemsets.4 <- apriori(Groceries, parameter=list(minlen=4, maxlen=4,
                                                support=0.02, target="frequent itemsets"))

summary(itemsets.4)

inspect(sort(itemsets.4, by ="support")) # nothing

# no 4-itemset satisfies the minimum support of 0.02. 
# If we lower down the minimum support to 0.007 then....

itemsets.4 <- apriori(Groceries, parameter=list(minlen=4, maxlen=4,
                                                support=0.007, target="frequent itemsets"))

summary(itemsets.4)
inspect(sort(itemsets.4, by ="support"))

# there are three frequent 4-itemsets if the minimum support is 0.007.


## # if the parameter maxlen is not specified, then....

itemsets<- apriori( Groceries , parameter = list( minlen=1,
                                                  support =0.02 , target ="frequent itemsets"))

summary( itemsets )
# this summarizes that: there are 59 frequent 1-itemsets; 
# 61 frequent 2-itemsets; and 2 frequent 3-itemsets

inspect(sort( itemsets , by ="support")) 
# this will rank the itemsets by their support, regardless of itemsets with 1 item or 2 items.
# row 17: {other vegetables, whole milk}  with support = 0.07483477

##  GETTING THE RULES instead of  FREQUENT ITEMSETS

rules <- apriori(Groceries, parameter=list(support=0.001,
                                           confidence=0.6, target = "rules")) #plotting itesmest with min suport = 0.001 and min confidence = 0.6

plot(rules) # scatter plot of all 2918 rules

# Scatter plot with custom measures and can add limiting the plot to the 100 with the 
# largest value for for the shading measure. 
plot(rules, measure = c("support", "confidence"), shading = "lift", col = "black")#, limit = 100) #can remove the )# #Whats the diff bw this and prev code - jst colour

#confidence, support and lift are like criteria for rules just like accuracy and others in other contexts

# more information about plot() under 'arules':
# http://127.0.0.1:23659/library/arulesViz/html/plot.html




# PLOT SOME TOP RULES FOR VISUALZATION:

# the top 3 rules sorted by LIFT:
inspect(head(sort(rules, by="lift"), 3)) #Sort by lift # Why not just plot lift?

# the top 5 rules sorted by LIFT
inspect(head(sort(rules, by="lift"), 5))
highLiftRules <- head(sort(rules, by="lift"), 5)

# plot the top 5 rules above for visualzation:
plot(highLiftRules, method="graph") # this is simple and a bit difficult to see

# more parameters added, plot looks better:
plot(highLiftRules, method = "graph", engine = "igraph",
     edgeCol = "blue", alpha = 1)
# alpha = c(0,1)
# the size of the node is sorted by the support.
# the darkness of the color represents the change in lift


plot(highLiftRules, method = "graph", engine = "igraph",
     nodeCol = "red", edgeCol = "blue", alpha = 1)
# this will fix the color be "red" for all lift values, 
# only the size of the node is sorted by the support.



#some common choices for 'method':
# matrix, mosaic, doubledecker, graph, paracoord, scatterplot, grouped matrix, two-key plot, matrix3D















