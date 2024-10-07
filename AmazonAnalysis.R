library(tidymodels)
library(embed)
library(vroom)

test <- vroom('test.csv')
train <- vroom('train.csv')

my_recipe <- recipe(ACTION ~., data=train) |>
step_mutate_at(all_numeric(), fn = factor) |>
  step_other(all_nominal_predictors(), threshold = .001) |> 
  step_dummy(all_nominal_predictors()) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)
