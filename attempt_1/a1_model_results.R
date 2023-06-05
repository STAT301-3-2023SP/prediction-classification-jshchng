# get final model results

library(tidymodels)
library(tidyverse)

tidymodels_prefer()

result_files <- list.files("attempt_1/results/", "*.rda", full.names = TRUE)

for(i in result_files){
  load(i)
}

#### Set up parallel processing
parallel::detectCores()
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

train <- read_csv("data/train.csv") %>%
  select(get_var, id, y) %>%
  mutate(x120 = as.factor(x120)) %>%
  mutate(y = as.factor(y))

test <- read_csv("data/test.csv") %>%
  select(get_var, id) %>%
  mutate(x120 = as.factor(x120))


##############################################
# baseline/null model
null_model <- null_model(mode = "classification") %>%
  set_engine("parsnip")

null_workflow <- workflow() %>%
  add_model(null_model) %>%
  add_recipe(recipe_sink)

null_fit <- null_workflow %>%
  fit_resamples(resamples = a1_folds,
                control = control_resamples(save_pred = TRUE,
                                            save_workflow = TRUE))
null_metrics <- null_fit %>%
  collect_metrics()
# roc_auc = .5, accuracy = .504

###############################################
# organzie results to find best overall
# put all tune_grids together

model_set <- as_workflow_set(
  "boosted tree" = bt_tune,
  "random forest" = rf_tune,
  "svm_poly" = svmp_tune,
  "svm_rbf" = svmr_tune,
  "mars" = mars_tune,
  "mlp" = mlp_tune
)


# plot of results
model_set <- model_set %>%
  group_by(wflow_id) %>%
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>%
  select(best) %>%
  unnest(cols = c(best))

model_results <- model_set %>%
  group_by(wflow_id) %>%
  mutate(best = map(result, show_best, metric = "roc_auc", n = 1)) %>%
  select(best) %>%
  unnest(cols = c(best)) %>%
  select(wflow_id, mean)

model_table <- model_set %>%
  autoplot(metric = "roc_auc", select_best = TRUE) +
  theme_minimal() +
  geom_text(aes(y = mean + .03, label = wflow_id), angle = 90, hjust = 1) +
  theme(legend.position = "none")

# 3 worst models
# mars, num_terms = 5, prod_degree = 1
# bt,mtry = 4, min_n = 11, learn_rate = 0.03981072
# rf, mtry = 1, min_n = 21

#### Best models
# svmp, cost = 32, degree = 3, scale_factor = 0.0005623413
# svmr, cost = 32, scale_factor = 0.003162278
# mlp, hidden_units = 3, penalty = 1

# extract workflows
bt_workflow <- extract_workflow(bt_tune)
rf_workflow <- extract_workflow(rf_tune)
mars_workflow <- extract_workflow(mars_tune)
svmp_workflow <- extract_workflow(svmp_tune)
svmr_workflow <- extract_workflow(svmr_tune)
mlp_workflow <- extract_workflow(mlp_tune)

################################### finalize the best 3 workflows
best_bt <- bt_workflow %>%
  finalize_workflow(select_best(bt_tune, metric = "roc_auc"))

best_mlp <- mlp_workflow %>%
  finalize_workflow(select_best(mlp_tune, metric = "roc_auc"))

best_rf <- rf_workflow %>%
  finalize_workflow(select_best(rf_tune, metric = "roc_auc"))

best_mars <- mars_workflow %>%
  finalize_workflow(select_best(mars_tune, metric = "roc_auc"))

best_svmr <- svmr_workflow %>%
  finalize_workflow(select_best(svmr_tune, metric = "roc_auc"))

best_svmp <- svmp_workflow %>%
  finalize_workflow(select_best(svmp_tune, metric = "roc_auc"))

# fit training to final workflows

bt_fit <- fit(best_bt, train)

rf_fit <- fit(best_rf, train)

mars_fit <- fit(best_mars, train)

svmp_fit <- fit(best_svmp, train)

svmr_fit <- fit(best_svmr, train)

mlp_fit <- fit(best_mlp, train)


save(model_results, file = "attempt_1/results/model_results.rda")
save(model_set, mars_fit, bt_fit, rf_fit, svmr_fit, svmp_fit, mlp_fit,
     file = "attempt_1/results/results.rda")

################################################
load("attempt_1/results/results.rda")

# fit test data
final_pred_bt <- predict(bt_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_mlp <- predict(mlp_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_rf <- predict(rf_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_mars <- predict(mars_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_svmp <- predict(svmp_fit, test) %>% 
  bind_cols(test %>% select(id))

final_pred_svmr <- predict(svmr_fit, test) %>% 
  bind_cols(test %>% select(id))

# untransform response variable
final_pred_bt <- final_pred_bt %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)

final_pred_mlp <- final_pred_mlp %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)

final_pred_rf <- final_pred_rf %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)

final_pred_mars <- final_pred_mars %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)

final_pred_svmr <- final_pred_svmr %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)

final_pred_svmp <- final_pred_svmp %>%
  mutate(y = .pred_class) %>%
  select(-.pred_class)


save(final_pred_bt, final_pred_rf, final_pred_mars, final_pred_svmp, final_pred_svmr, final_pred_mlp,
     file = "attempt_1/results/final_pred.rda")
