#Bike share

library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)

bikeData <- vroom("trainBikeShare.csv")
dplyr::glimpse(bikeData)
skimr::skim(bikeData)
plot_intro(bikeData)
plot_histogram()
plot_correlation(bikeData)

#Key features: temp, atemp, humidity, season, weather, windspeed

tempPlot <- ggplot(data=bikeData, mapping = aes(x=temp,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
weatherPlot <- ggplot(data=bikeData, mapping=aes(x=weather)) +
  geom_bar()
weatherPlot
humidPlot <- ggplot(data=bikeData, mapping = aes(x=humidity,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
windPlot <- ggplot(data=bikeData, mapping = aes(x=windspeed,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
(tempPlot + weatherPlot) / (humidPlot + windPlot)
