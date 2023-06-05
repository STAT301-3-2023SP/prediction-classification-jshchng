# RF tuning

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
rf_model <- rand_forest(mode = "classification",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger", importance = "impurity")

rf_params <- extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(c(1, 13))) 

# grid
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_sink)

#### stop cluster and start timer

rf_tune <- tune_grid(rf_workflow,
                     resamples = a1_folds,
                     grid = rf_grid,
                     control = control_grid(save_pred = TRUE,
                                            save_workflow = TRUE,
                                            parallel_over = "everything"))

stopCluster(cl) 

# save 
save(rf_tune, file = "attempt_1/results/tuned_rf.rda")
