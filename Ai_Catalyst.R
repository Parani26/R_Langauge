set.seed(1)
setwd("/Users/tech26/Desktop/ai catalyst")
flood <- read.csv("AEGISDataset.csv")
head(flood)
dim(flood)
flood.intensity <- flood$flood_heig

# Displaying data points
hist(flood_heig, freq=FALSE, main = paste(""),
     xlab = "Flood Intensities", ylab="Proportion", col = "blue")

hist(elevation, freq=FALSE, main = paste(""),
     xlab = "Elevations", ylab="Proportion", col = "red")

hist(Precipitation, freq=FALSE, main = paste(""),
     xlab = "Precipitations", ylab="Proportion", col = "green")

#Standardising data
flood_copy <- flood
flood_subset <- flood_copy[, c("elevation", "precipitat")]
flood_subset <- scale(flood_subset)
flood_standard <- data.frame(
  elevation = flood_subset[, "elevation"],
  precipitat = flood_subset[, "precipitat"]
)
Elevation.s <- flood_standard$elevation
Precipitation.s <-flood_standard$precipitat

#Carrying out linear regression
regression <- lm(flood.intensity ~ Elevation.s + Precipitation.s, data = flood_standard)

# Model is signifcant
summary(regression)
cor(flood.intensity, Elevation.s)
cor(flood.intensity, Precipitation.s)

#Predicting data point
new = data.frame(Elevation.s = 0.5, Precipitation.s = 0.3) 
predict(regression, newdata = new) 

