# 04 — Evaluation, Metrics, and Reliable Experiments

A model is useful only to the extent that its measured performance reflects the task where it will be used. Evaluation is not a final decoration; it influences data collection, model choice, deployment risk, and maintenance.

## Train, validation, and test

- **Training data** is used to fit parameters.
- **Validation data** helps choose hyperparameters and compare development options.
- **Test data** is reserved for the final assessment after development choices are complete.

For a small dataset, cross-validation can make more efficient use of available examples, but it does not remove every source of bias. If observations are grouped by person, device, location, or time, the split should respect that structure when it matters to the deployment setting.

## Regression metrics

Use the companion script at ../03-machine-learning/metrics.py for small, readable implementations.

- **MAE:** average absolute error. Easy to interpret in the target's units and less sensitive to large residuals than squared-error measures.
- **MSE:** average squared error. Penalizes large residuals strongly and uses squared target units.
- **RMSE:** square root of MSE. Returns to the target's units but still emphasizes large errors.
- **R-squared:** compares residual error with variation around the target mean. It can be negative on evaluation data, and it is undefined for a constant target under the standard formula.

No metric is universally best. Consider the cost of different mistakes and the scale of the target.

## Classification metrics

- **Accuracy:** fraction of all predictions that are correct.
- **Precision:** fraction of predicted positives that are actually positive.
- **Recall:** fraction of actual positives that are found.
- **F1:** harmonic mean of precision and recall.
- **Confusion matrix:** counts true positives, false positives, true negatives, and false negatives.
- **ROC-AUC:** measures ranking discrimination across classification thresholds; for severe imbalance, precision-recall analysis may also be informative.

For rare events, a model that always predicts the majority class can have high accuracy but zero useful recall for the rare class.

## Common leakage traps

1. Scaling or imputing the whole dataset before splitting.
2. Selecting features using target information from the test set.
3. Randomly splitting time-series rows when future information can leak into the past.
4. Putting near-duplicate observations into both train and test partitions.
5. Fitting text vocabulary or feature selection on evaluation examples when the intended workflow should learn those only from training data.
6. Repeatedly choosing models based on test results.

A safe pattern is to split first, fit each learned preprocessing step on training data, and apply the fitted transformation to validation and test data.

## A compact experiment protocol

For every comparison, record:

- the question and hypothesis;
- data source and split strategy;
- baseline and model configurations;
- metric definition and evaluation set;
- random seed and relevant software versions;
- results, uncertainty where possible, and observed failure cases;
- limitations and what experiment you would run next.

## Exercises

1. Construct a rare-positive dataset where accuracy looks good but recall is poor.
2. Explain why RMSE and MAE can rank two models differently.
3. Give a time-series example where random splitting creates an unrealistic evaluation.
4. Identify which preprocessing operations should be fit only on training data.
5. If you try 100 model configurations and pick the best test score, why might that score be optimistic?
