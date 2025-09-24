#Bike share

library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(DataExplorer)
library(GGally)
library(patchwork)
library(rpart)
library(ranger)

#Set up training data

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
# Initial graphs

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


#Recipe for regression with penalties


bike_recipe <- recipe(count~., data= bikeData) %>%
  step_mutate(weather= ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather)) %>%
  step_time(datetime,features=c("hour")) %>%
  step_mutate(season = factor(season)) %>%
  step_corr(all_numeric_predictors(), threshold = .5) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped <- prep(bike_recipe)
bake(prepped, new_data=bikeData)

preg_model <- linear_reg(penalty=tune(),mixture=tune()) %>%
  set_engine("glmnet")

preg_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(preg_model)

tuning_grid <- grid_regular(penalty(), mixture(), levels = 5)

folds <- vfold_cv(bikeData, v = 5, repeats=1)

CV_results <- preg_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse,mae))

collect_metrics(CV_results) %>%
  filter(.metric=="rmse") %>%
  ggplot(data=.,aes(x=penalty, y= mean, color=factor(mixture))) +
  geom_line()

bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <- 
  preg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=bikeData)

preg_preds <- predict(final_wf, new_data = testData)


#Regression tree


reg_tree <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n=tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

tree_workflow <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(reg_tree)

tree_grid <- grid_regular(tree_depth(), cost_complexity(), min_n(), levels=5)

folds <- vfold_cv(bikeData, v = 5, repeats=1)

tree_results <- tree_workflow %>%
  tune_grid(resamples = folds, grid=tree_grid, metrics=metric_set(rmse,mae))

tree_bestTune <- tree_results %>%
  select_best(metric="rmse")

final_tree_wf <-
  tree_workflow %>%
  finalize_workflow(tree_bestTune) %>%
  fit(data=bikeData)

tree_preds <-predict(final_tree_wf, new_data = testData)


# Random Forest


forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),trees=500) %>%
  set_engine("ranger") %>%
  set_mode("regression")

forest_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(forest_mod)

forest_grid <- grid_regular(mtry(range=c(1, 11)),
                       min_n(),
                       levels=5)

for_folds <- vfold_cv(bikeData, v = 5, repeats = 1)

forest_results <- forest_wf %>%
  tune_grid(resamples= for_folds, grid= forest_grid, metrics=metric_set(rmse, mae))

forest_bestTune <- forest_results %>%
  select_best(metric = "rmse")

final_forest_wf <- forest_wf %>%
  finalize_workflow(forest_bestTune) %>%
  fit(data=bikeData)

forest_preds <- predict(final_forest_wf, new_data=testData)


#Boosting




kaggle_submission <- forest_preds %>%
  bind_cols(., testData) %>%
  mutate(.pred = exp(.pred)) %>%
  select(datetime, .pred) %>%
  rename(count =.pred) %>%
  mutate(count = pmax(0, count)) %>%
  mutate(datetime = as.character(format(datetime)))

vroom_write(x=kaggle_submission, file="./ForestPreds.csv", delim=",")


