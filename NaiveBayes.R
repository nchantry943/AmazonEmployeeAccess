library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)

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

## Naive Bayes
nb_mod <- naive_Bayes(Laplace = tune(), smoothness = tune()) |>
  set_mode('classification') |>
  set_engine('naivebayes')

nb_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(nb_mod)

tuning_grid <- grid_regular(Laplace(range = c(.01, 100)),
                            smoothness(range = c(.01, 5)),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)

CV_results <- nb_wf |>
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

best <- CV_results |>
  select_best()

final_wf_nb <- nb_wf |>
  finalize_workflow(best) |>
  fit(data = train)

nb_preds <- predict(final_wf_nb, new_data = test, type = 'prob')

preds_nb <- nb_preds |>
  bind_cols(test) |>
  rename(ACTION = .pred_1) |>
  select(id, ACTION)

vroom_write(x=preds_nb, file = "./NBPredsPCA.csv", delim=",")
