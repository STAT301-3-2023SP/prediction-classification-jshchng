library(tidyverse)
library(tidymodels)

tidymodels_prefer()

load("attempt_1/results/var_importance.rda")

train_data <- read_rds("data/train_data.rds") %>%
  select(get_var, id, y) %>%
  mutate(x120 = as.factor(x120))

# fold data
a1_folds <- vfold_cv(train_data, v = 5, repeats = 3, strata = y)

# sample recipe 
recipe_sink <- recipe(y ~ ., data = train_data) %>%
  step_rm(id) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_impute_knn(all_predictors()) %>%
  step_normalize(all_predictors())

# prep and bake
#recipe1 %>%
# prep(train_data) %>%
# bake(new_data = NULL) %>%
# view()

save(a1_folds, recipe_sink, file = "attempt_1/results/a1_info.rda")
