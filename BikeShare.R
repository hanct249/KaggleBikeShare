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
  select(-c(casual,registered)) %>%
  mutate(count=log(count))

str(bikeData$count)

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

bike_recipe <- recipe(count~., data= bikeData) %>%
  step_mutate(weather= ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime,features=c("hour")) %>%
  step_mutate(season = factor(season)) %>%
  step_corr(all_numeric_predictors(), threshold = .1) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors())

  


my_linear_model <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

bike_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(my_linear_model) %>%
  fit(data=bikeData)

lin_preds <- predict(bike_workflow, new_data = testData)


kaggle_submission <- lin_preds %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")

