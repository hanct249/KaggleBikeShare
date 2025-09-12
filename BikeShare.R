#Bike share

library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)

bikeData <- vroom("trainBikeShare.csv")
bikeData <- bikeData %>%
  select(-c(casual,registered))
testData <- vroom("testBikeShare.csv")
dplyr::glimpse(bikeData)
skimr::skim(bikeData)
plot_intro(bikeData)
plot_histogram(bikeData)
plot_correlation(bikeData)

#Key features: temp, atemp, humidity, season, weather, windspeed

tempPlot <- ggplot(data=bikeData, mapping = aes(x=temp,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
weatherPlot <- ggplot(data=bikeData, mapping=aes(x=weather)) +
  geom_bar()
humidPlot <- ggplot(data=bikeData, mapping = aes(x=humidity,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
windPlot <- ggplot(data=bikeData, mapping = aes(x=windspeed,y=count)) +
  geom_point() + 
  geom_smooth(se=FALSE)
(tempPlot + weatherPlot) / (humidPlot + windPlot)

my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") %>%
  fit(formula=count~.-datetime, data=bikeData)

bike_predictions <- predict(my_linear_model, new_data=testData)
bike_predictions

kaggle_submission <- bike_predictions %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

