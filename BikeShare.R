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
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped <- prep(bike_recipe)
bake(prepped, new_data=bikeData)


preg_model <- linear_reg(penalty=.1,mixture=0) %>%
  set_engine("glmnet")
preg_model2 <- linear_reg(penalty=.5,mixture=0) %>%
  set_engine("glmnet")
preg_model3 <- linear_reg(penalty=.1,mixture=1) %>%
  set_engine("glmnet")
preg_model4 <- linear_reg(penalty=.5,mixture=1) %>%
  set_engine("glmnet")
preg_model5 <- linear_reg(penalty=.1,mixture=0.5) %>%
  set_engine("glmnet")

preg_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bikeData)
preg_workflow2 <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model2) %>%
  fit(data=bikeData)
preg_workflow3 <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model3) %>%
  fit(data=bikeData)
preg_workflow4 <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model4) %>%
  fit(data=bikeData)
preg_workflow5 <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model5) %>%
  fit(data=bikeData)

lin_preds <- predict(preg_workflow, new_data = testData)
lin_preds2 <- predict(preg_workflow2, new_data = testData)
lin_preds3 <- predict(preg_workflow3, new_data = testData)
lin_preds4 <- predict(preg_workflow4, new_data = testData)
lin_preds5 <- predict(preg_workflow5, new_data = testData)


kaggle_submission <- lin_preds %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))
kaggle_submission2 <- lin_preds2 %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))
kaggle_submission3 <- lin_preds3 %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))
kaggle_submission4 <- lin_preds4 %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))
kaggle_submission5 <- lin_preds5 %>%
  bind_cols(., testData) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))


vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
vroom_write2(x=kaggle_submission2, file="./LinearPreds2.csv", delim=",")
vroom_write3(x=kaggle_submission3, file="./LinearPreds3.csv", delim=",")
vroom_write4(x=kaggle_submission4, file="./LinearPreds4.csv", delim=",")
vroom_write5(x=kaggle_submission5, file="./LinearPreds5.csv", delim=",")

