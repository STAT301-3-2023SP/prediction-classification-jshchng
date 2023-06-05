# MARS tuning

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
mars_model <- mars(mode = "classification", num_terms = tune(), prod_degree = tune()) %>%
  set_engine("earth")

mars_params <- extract_parameter_set_dials(mars_model)

?num_terms

# grid
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow
mars_workflow <- workflow() %>%
  add_model(mars_model) %>%
  add_recipe(recipe_sink)

#### stop cluster and start timer

mars_tune <- tune_grid(mars_workflow,
                       resamples = a1_folds,
                       grid = mars_grid,
                       control = control_grid(save_pred = TRUE,
                                              save_workflow = TRUE,
                                              parallel_over = "everything"))
# end timer

stopCluster(cl) 

# save 
save(mars_tune, file = "attempt_1/results/tuned_mars.rda")
