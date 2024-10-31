library(tidymodels)
library(vroom)
library(embed)
library(kknn)

test <- vroom('test.csv')
train <- vroom('train.csv')

train$ACTION <- factor(train$ACTION)

# my_recipe <- recipe(ACTION ~., data=train) |>
#   step_mutate_at(all_numeric(), fn = factor) |>
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
#   step_normalize(all_numeric_predictors())

my_recipe <- recipe(ACTION ~., data=train) |>
  step_mutate_at(all_numeric(), fn = factor) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_numeric_predictors()) |>
  step_pca(all_predictors(), threshold = .8)

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

## KNN Analysis
rf_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 1000) |>
  set_engine('ranger') |>
  set_mode('classification')

rf_wf <- workflow () |>
  add_recipe(my_recipe) |>
  add_model(rf_mod) 

tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(range = c(2, 10)),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- rf_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_rf <- rf_wf |>
  finalize_workflow(best) |>
  fit(data = train)

rf_preds <- predict(final_wf_rf, new_data = test, type = 'prob')

preds_rf <- rf_preds |>
  bind_cols(test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(x=preds_rf, file = "./RFPredsPCA.csv", delim=",")
