# KNN tuning

library(tidyverse)
library(tidymodels)
library(tictoc)
library(doParallel)

tidymodels_prefer()

# load
load("attempt_1/results/a1_info.rda")

#### Set up parallel processing
parallel::detectCores()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

#### Define model engine and workflow
knn_model <- nearest_neighbor(mode = "classification", neighbors = tune()) %>%
  set_engine("kknn")

knn_params <- extract_parameter_set_dials(knn_model)

# grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe_sink)

#### stop cluster 
knn_tune <- tune_grid(knn_workflow,
                      resamples = a1_folds,
                      grid = knn_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything"))

stopCluster(cl) 

# save 
save(knn_tune, file = "attempt_1/results/tuned_knn.rda")
