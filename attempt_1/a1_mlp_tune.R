# MLP tuning

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
mlp_model <- mlp(mode = "classification", hidden_units = tune(), penalty = tune()) %>%
  set_engine("nnet")

mlp_params <- extract_parameter_set_dials(mlp_model)

# grid
mlp_grid <- grid_regular(mlp_params, levels = 5)

# workflow
mlp_workflow <- workflow() %>%
  add_model(mlp_model) %>%
  add_recipe(recipe_sink)


mlp_tune <- tune_grid(mlp_workflow,
                      resamples = a1_folds,
                      grid = mlp_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything"))


stopCluster(cl) 

# save 
save(mlp_tune, file = "attempt_1/results/tuned_mlp.rda")

