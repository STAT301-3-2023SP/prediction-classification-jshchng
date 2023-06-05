# SVMP tuning

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
svmp_model <- 
  svm_poly(cost = tune(), degree = tune(), scale_factor = tune()) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

svmp_params <- extract_parameter_set_dials(svmp_model) 


# grid
svmp_grid <- grid_regular(svmp_params, levels = 5)

# workflow
svmp_workflow <- workflow() %>%
  add_model(svmp_model) %>%
  add_recipe(recipe_sink)

svmp_tune <- tune_grid(svmp_workflow,
                       resamples = a1_folds,
                       grid = svmp_grid,
                       control = control_grid(save_pred = TRUE,
                                              save_workflow = TRUE,
                                              parallel_over = "everything"))


stopCluster(cl) 

# save 
save(svmp_tune, file = "attempt_1/results/tuned_svmp.rda")
