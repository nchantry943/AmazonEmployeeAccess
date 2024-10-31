library(tidymodels)
library(embed)
library(vroom)

test <- vroom('test.csv')
train <- vroom('train.csv')

train$ACTION <- factor(train$ACTION)

# my_recipe <- recipe(ACTION ~., data=train) |>
#   step_mutate_at(all_numeric(), fn = factor) |>
#   step_other(all_nominal_predictors(), threshold = .01) |> 
#   step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
#   step_normalize(all_nominal_predictors())

my_recipe <- recipe(ACTION ~., data=train) |>
  step_mutate_at(all_numeric(), fn = factor) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_numeric_predictors()) |>
  step_pca(all_predictors(), threshold = .8)


prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


## Logistic Regression
log_reg_mod <- logistic_reg() |>
  set_engine('glm')

workflow_log <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(log_reg_mod) |>
  fit(data = train)

log_reg_pred <- predict(workflow_log, new_data = test, type = 'prob')

preds_logistic <- log_reg_pred |>
  bind_cols(test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(x=preds_logistic, file = "./LogisticPredsPCA.csv", delim=",")

## Penalized Regression
pen_mod <- logistic_reg(mixture = tune(), penalty = tune()) |>
  set_engine('glmnet')

pen_workflow <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(pen_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- pen_workflow |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_pen <- pen_workflow |>
  finalize_workflow(best) |>
  fit(data = train)

pen_pred <- final_wf_pen |> predict(new_data = test, type = 'prob')

preds_penalized <- pen_pred |>
  bind_cols(test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(x=preds_penalized, file = "./PenalizedLogisticPredsPCA.csv", delim=",")

