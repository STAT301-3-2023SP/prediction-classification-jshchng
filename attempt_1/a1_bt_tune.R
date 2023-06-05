# BT tuning

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
bt_model <- boost_tree(mode = "classification",
                       min_n = tune(), 
                       mtry = tune(), 
                       learn_rate = tune()) %>%
  set_engine("xgboost", imporatnce = "impurity")

# params
bt_params <- extract_parameter_set_dials(bt_model) %>%
  update(learn_rate = learn_rate(c(-5, -0.2)),
         mtry = mtry(c(1,13))) 

# grid
bt_grid <- grid_regular(bt_params, levels = 5)

# workflow
bt_workflow <- workflow() %>%
  add_model(bt_model) %>%
  add_recipe(recipe_sink)

#### stop cluster
bt_tune <- tune_grid(bt_workflow,
                     resamples = a1_folds,
                     grid = bt_grid,
                     control = control_grid(save_pred = TRUE,
                                            save_workflow = TRUE,
                                            parallel_over = "everything"))

stopCluster(cl) 

# save 
save(bt_tune, file = "attempt_1/results/tuned_bt.rda")
