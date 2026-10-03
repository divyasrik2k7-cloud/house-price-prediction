# Methodology

1. **Dataset:** California Housing dataset provided by scikit-learn.
2. **Inspection:** Review feature names, target, data types, and basic descriptive statistics.
3. **Split:** Hold out 20% for testing with `random_state=42`.
4. **Preprocessing:** Median imputation is placed inside each model pipeline. This ensures any learned imputation values are fit only on training data.
5. **Models:** Linear Regression, Decision Tree Regression, and Random Forest Regression.
6. **Evaluation:** Compare models on the same held-out test set using MAE, MSE, RMSE, and R².
7. **Visualization:** Generate a target distribution, correlation heatmap, actual-vs-predicted plots, and residual plots.
8. **Reproducibility:** Fixed random seeds are used for splitting and tree-based models.

## Metrics
- **MAE:** Mean absolute prediction error; lower is better.
- **MSE:** Mean squared prediction error; lower is better.
- **RMSE:** Square root of MSE, in the target's units; lower is better.
- **R²:** Proportion of target variance explained relative to a mean baseline; values closer to 1 indicate better fit, while negative values are possible.

## Limitation
This dataset is a historical educational dataset. Predictions should not be treated as current property appraisals or financial advice.
