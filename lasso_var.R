library(tidymodels)
library(tidyverse)
library(glmnet)
library(doParallel)

tidymodels_prefer()

train <- read_csv("data/train.csv") %>%
  mutate(y = as.factor(y))

set.seed(3013)
fold_var_select <- vfold_cv(train, folds = 5, repeats = 1, strata = y)

#### Set up parallel processing
parallel::detectCores()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

# Assuming 'data' is your dataset and 'response' is the response variable
data_recipe <- recipe(y ~ ., data = train) %>%
  step_rm(id) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_corr(all_predictors())

lasso_model <- logistic_reg(mode = "classification", penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

lasso_params <- extract_parameter_set_dials(lasso_model) %>%
  update(penalty = penalty(range = c(0.02, 0.1), trans = NULL))


lasso_grid <- grid_regular(lasso_params, levels = 5)

lasso_workflow <- workflow() %>%
  add_recipe(data_recipe) %>%
  add_model(lasso_model)

lasso_tune <- tune_grid(
  lasso_workflow, 
  resamples = fold_var_select,
  grid = lasso_grid,
  control = control_grid(parallel_over = "everything")
)


#stop
stopCluster(cl) 

# get tune results
lasso_final_workflow <- lasso_workflow %>%
  finalize_workflow(select_best(lasso_tune, metric = "roc_auc"))



# fit final workflow to entire train data

lasso_fit <- fit(lasso_final_workflow, train)

# get coefficients
lasso_var <- lasso_fit %>%
  tidy()

get_var <- lasso_var %>%
  filter(estimate != 0, term != "(Intercept)") %>%
  pull(term)

save(get_var, file = "attempt_1/results/var_importance.rda")
