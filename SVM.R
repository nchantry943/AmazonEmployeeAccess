library(tidymodels)
library(vroom)
library(embed)
library(kernlab)

test <- vroom('test.csv')
train <- vroom('train.csv')

train$ACTION <- factor(train$ACTION)

my_recipe <- recipe(ACTION ~., data=train) |>
  step_mutate_at(all_numeric(), fn = factor) |>
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)


svm_mod <- svm_rbf(cost = tune(), rbf_sigma = tune()) |>
  set_mode('classification') |>
  set_engine('kernlab')

svm_wf <- workflow () |>
  add_recipe(my_recipe) |>
  add_model(svm_mod) 

tuning_grid <- grid_regular(rbf_sigma(range = c(.01, 10)),
                            cost(range = c(.01, 10)),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- svm_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_svm <- svm_wf |>
  finalize_workflow(best) |>
  fit(data = train)

svm_preds <- predict(final_wf_svm, new_data = test, type = 'prob')

preds_svm <- svm_preds |>
  bind_cols(test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(x=preds_svm, file = "./SVMPreds.csv", delim=",")
save(file = "./SVMPreds.csv")
