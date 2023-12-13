# Building a Ratings Prediction Model

A Data Science project about model building for DSC80 at UCSD.

**Authors:** Catherine Back, Lorenzo Ramos

[Exploratory Data Anaylysis](https://ramosrenzo.github.io/Recipes-Research/)

---

## Framing the Problem
We plan on building a regression model to predict the amount of calories a recipe will have based on other features such as `minutes`, `n_steps`, `n_ingredients`, and more.  The calories a recipe can have varies from 0-infinity; thus why we will be performing a regression task instead of a classification task.

**Response Variable:**  For our response variable, we selected calories, a continuous quantitative variable, making a regression task viable.  Correctly predicting the `calories` of a recipe may be important to people who like to keep track of their nutritional values day to day.  By building this model, we could help these people identify what makes a recipes `calories` high or low.

**Metrics:**  Due to our model being a regressor, we will be unable to use metrics such as accuracy, precision, recall, or f1 score.  So, we will be evaluating metrics such as **R<sup>2</sup>** and **RMSE** (Root Mean Squared Error).  The metric R<sup>2</sup> will give us a statistical measure of how well predicted values from the model match the actual values in the original dataset, while the metric RMSE will measure the average magnitude of the errors between these predicted and actual values.

**Known Information:** At the time of prediction, the only known information we would have is `n_steps` and `n_ingredients`.  This information is given to us due to the fact that each recipe needs to have a set number of steps and a set number of ingredients.  Users who add recipes have to explicitly state the steps and ingredients they use in order to make, so it's implied that these features are known at the time of prediction, in order for people to follow the recipe as closely as possible.

---

## Baseline Model
For our baseline model, we intend to use 2 features, `n_steps` and `n_ingredients`, which are both ordinal discrete variables.  We are using these two features because in our heatmap, it was two of the variables that didn’t have a negative correlation with `calories`.

**Quantitative Features:** `n_steps`, `n_ingredients`

**Feature Transformations:** We will only apply a square transformation using FunctionTransformer to both `n_steps` and `n_ingredients`.

**Performance:** 

| Metric   |    Value |
|:---------|---------:|
| R²       | 0.994159 |
| RMSE     | 6.85192  |

## Final Model

> New Features Added

**Quantitative Variable(s):** `minutes`, `total_fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated_fat (PDV)`, `carbohydrates (PDV)`

All of the features we are adding are all quantitative variables.  The reason we decided to add these features is because looking back on our heatmap, `minutes` doesn’t have a negative correlation and the nutritional labels have fairly good correlation with `calories`.

| Metric   |    Value |
|:---------|---------:|
| R²       | 0.994145 |
| RMSE     | 6.85602  |

---

## Fairness Analysis