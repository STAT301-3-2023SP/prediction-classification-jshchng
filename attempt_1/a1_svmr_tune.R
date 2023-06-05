# SVMR tuning

library(tidyverse)
library(tidymodels)
library(tictoc)
library(doParallel)

tidymodels_prefer()

# load
load("attempt_1/results/a1_info.rda")

#### Set up parallelcessing
parallel::detectCores()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

#### Define model engine and workflow
svmr_model <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svmr_params <- extract_parameter_set_dials(svmr_model) 


# grid
svmr_grid <- grid_regular(svmr_params, levels = 5)

# workflow
svmr_workflow <- workflow() %>%
  add_model(svmr_model) %>%
  add_recipe(recipe_sink)


svmr_tune <- tune_grid(svmr_workflow,
                       resamples = a1_folds,
                       grid = svmr_grid,
                       control = control_grid(save_pred = TRUE,
                                              save_workflow = TRUE,
                                              parallel_over = "everything"))

stopCluster(cl) 

# save 
save(svmr_tune, file = "attempt_1/results/tuned_svmr.rda")
