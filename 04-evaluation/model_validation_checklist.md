# Model Validation Checklist

Use this checklist before accepting a model result, comparing experiments, or deploying a model.

## Problem and target

- [ ] The prediction target is explicitly defined.
- [ ] The time at which each prediction is made is clear.
- [ ] Inputs are available at that time in the real workflow.
- [ ] The baseline reflects a meaningful simple alternative.
- [ ] The metric reflects the cost of different errors.

## Data and split

- [ ] The source and version of the dataset are recorded.
- [ ] Duplicate and related examples cannot accidentally cross the split boundary.
- [ ] Groups such as people, devices, or sites are separated when required.
- [ ] The split reflects the deployment scenario.
- [ ] Time-based problems use a time-respecting evaluation design.
- [ ] Class imbalance and missingness have been inspected.

## Preprocessing

- [ ] Imputation is fitted using training data only.
- [ ] Scaling and normalization are fitted using training data only.
- [ ] Feature selection does not use test labels.
- [ ] Vocabulary, embeddings, or derived statistics do not leak held-out information.
- [ ] The exact preprocessing pipeline used during evaluation matches intended inference.

## Training and model selection

- [ ] Hyperparameters are chosen without using the final test set repeatedly.
- [ ] Random seeds and important environment details are recorded.
- [ ] Training and validation curves or diagnostics are inspected.
- [ ] A simple baseline is included.
- [ ] Failures, not just average scores, are reviewed.

## Reporting

- [ ] The number of test examples is reported.
- [ ] Metric definitions are stated.
- [ ] Units and aggregation choices are clear.
- [ ] Uncertainty or variation across seeds/splits is considered where appropriate.
- [ ] Subgroup performance is checked when meaningful and lawful.
- [ ] Limitations and expected distribution shifts are documented.

## Deployment readiness

- [ ] Inputs are validated.
- [ ] Missing or out-of-range features are handled deliberately.
- [ ] Latency and memory have been measured on realistic hardware.
- [ ] Monitoring and rollback plans exist for production use.
- [ ] Sensitive data and access permissions are handled correctly.

Passing the checklist does not prove a model is safe or useful. It makes common weaknesses easier to discover and document.
