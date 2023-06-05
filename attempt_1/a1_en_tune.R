# EN tuning

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
net_model <- logistic_reg(mode = "classification", 
                        penalty = tune(),
                        mixture = tune()) %>%
  set_engine("glmnet")

net_params <- extract_parameter_set_dials(net_model)

# grid
net_grid <- grid_regular(net_params, levels = 5)


net_workflow <- workflow() %>%
  add_model(net_model) %>%
  add_recipe(recipe_sink)


#### stop cluster and start timer
net_tune <- tune_grid(net_workflow,
                      resamples = a1_folds,
                      grid = net_grid,
                      control = control_grid(save_pred = TRUE,
                                             save_workflow = TRUE,
                                             parallel_over = "everything"))

stopCluster(cl) 

# save 
save(net_tune, file = "../attempt_1/results/tuned_net.rda")