# Building a Ratings Prediction Model

A Data Science project about model building for DSC80 at UCSD.

**Authors:** Catherine Back, Lorenzo Ramos

[Exploratory Data Anaylysis](https://ramosrenzo.github.io/Recipes-Research/)

---

## Framing the Problem

We plan to build a regression model to predict the number of calories a recipe will have based on features such as `minutes`, `n_steps`, `n_ingredients`, and more.  The calories a recipe can have vary from 0-infinity; thus we will be performing a regression task instead of a classification task.

**Response Variable:**  For our response variable, we selected `calories`, a continuous quantitative variable, making a regression task viable.  Correctly predicting the calories of a recipe may be important to people who like to keep track of their nutritional values day to day.  By building this model, we could help these people identify what makes a recipe's calories high or low.

**Metrics:**  Due to our model being a regressor, we will be unable to use metrics such as accuracy, precision, recall, or f1 score.  So, we will be evaluating metrics such as **R<sup>2</sup>** and **RMSE** (Root Mean Squared Error).  The metric R<sup>2</sup> will give us a statistical measure of how well-predicted values from the model match the actual values in the original dataset, while the metric RMSE will measure the average magnitude of the errors between these predicted and actual values.

**Known Information:** At the time of prediction, the only known information we would have is `n_steps` and `n_ingredients`.  This information is given to us since each recipe needs to have a set number of steps and a set number of ingredients.  Users who add recipes have to explicitly state the steps and ingredients they use to make, so it's implied that these features are known at the time of prediction, for people to follow the recipe as closely as possible.

---

## Baseline Model

For our baseline model, we intend to use 2 features, `n_steps` and `n_ingredients`, which are both ordinal discrete variables.  We are using these two features because in our heatmap, these were two of the variables that didn’t have a negative correlation with `calories`.  Thus, we will be using ‘n_steps’ and ‘n_ingredients’ as features for our baseline **Linear Regression** model.

**Quantitative Feature(s):** `n_steps`, `n_ingredients`

**Feature Transformations:** We will only apply a square transformation using FunctionTransformer to both `n_steps` and `n_ingredients`.  The reason for this is that when plotting the distributions of n_steps and n_ingredients, we can see that both distributions have a moderate right-tail skew.  A square transformation should transform this skewed data to conform to normality.

**Performance:** Using a Linear Regression algorithm for our baseline model, we can see that it’s not very good at predicting calories when compared to the actual calories values.  Our R<sup>2</sup> value was 0.0399, which implies that only about 4% of the variability in the response variable can be explained by the independent variables in the model.  Additionally, the RMSE for our baseline model is valued at 24.5335.  This is a measure of the average difference between the predicted and actual values, meaning that our RMSE is not nearly low enough to guarantee an accurate model.  Below we have provided a table for our metrics, for easier readability.

| Metric   |      Value |
|:---------|-----------:|
| R²       |  0.0399379 |
| RMSE     | 24.5335    |

---

## Final Model

> New Features Added

**Quantitative Feature(s):** `minutes`, `total_fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated_fat (PDV)`, `carbohydrates (PDV)`

All of the features we are adding are quantitative variables.  The reason we decided to add these features is that looking back on our heatmap, `minutes` doesn’t have a negative correlation, and the nutritional labels have a fairly good correlation with `calories`.

**Feature Transformations:** As before in our baseline model, we applied a square transformation using a Function Transformer on the columns `n_steps`, `n_ingredients` as well as `minutes`. We applied this transformation because after plotting the data points we saw a skew in the data and this conforms the data to normality. 
We also applied a standard scaler to the columns, `n_steps`, `n_ingredients`, `minutes`, `total_fat (PDV)`, `sugar (PDV)`, `sodium (PDV)`, `protein (PDV)`, `saturated_fat (PDV)`, and `carbohydrates (PDV)`. This was applied because a standard scaler transformation helps deal with multiple numbers on different scales efficiently. 
Finally, we applied a quantile transformation to the `minutes` columns with 5 different quantiles to normalize the distribution since the original was not normal.

**Modeling Algorithm:** Again, we went with Linear Regression like our base model.  By finding the best hyperparameters and using those for our final model, we are hoping to prevent overfitting/underfitting in our prediction model.  We performed a 5-fold cross-validation on these parameters:

```
   hyperparams = {
   'regressor__n_jobs': [1, 10, 20, -1],
   'regressor__fit_intercept': [True, False],
   'regressor__copy_X': [True, False],
   'regressor__positive': [True, False]
}
```

And through the use of GridSearch we found that the best parameters were:

```
{
 'regressor__copy_X': True,
 'regressor__fit_intercept': True,
 'regressor__n_jobs': 1,
 'regressor__positive': True
}
```

**Performance:** After applying the hyperparameters that were the best according to  GridSearch, our model's performance increased from an R<sup>2</sup> of 0.0399 to an R<sup>2</sup> of 0.9942. Our RMSE also improved from 24.5335 to 6.8533.  This improvement is attributed to many factors that were applied to our final model such as the addition of many columns from the nutritional facts and the different transformations applied. This model also improved due to GridSearch finding the best hyperparameters for our model.  Below we have a table for our metrics and visualization our final model's predicted calories vs actual calories performance.

| Metric   |    Value |
|:---------|---------:|
| R²       | 0.994154 |
| RMSE     | 6.85328  |

<iframe src="assets/calories_plot.html" width=800 height=600 frameBorder=0></iframe>

---

## Fairness Analysis
In the dataset of recipes, we split the recipes based on the amount of calories with less than 500 calories as low in calories and recipes with equal to or more than 500 calories as high in calories.

**Null Hypothesis:** Our model is fair in predicting recipe calories. Its precision for recipes with higher calories and lower calories are the same and any difference is due to random chance.

**Alternative Hypothesis:** Our model is unfair in predicting recipe calories. Its precision for higher calories is higher than its precision for recipes with lower calories. 

**Test Statistic:** The test statistic used is the difference in accuracy. Lower values of the test statistic point towards the null hypothesis, and higher values of the test statistic point towards the alternative.

**Observed Test Statistic:** 0.0256

**Method & Conclusion:** We performed a permutation test 500 times. With a calculated **p-value** of 1.0 and a **significance level** of 0.05, we fail to reject the null hypothesis.  Therefore, this **suggests** that there might be a similar estimation in both recipes that are low and high in calories.
