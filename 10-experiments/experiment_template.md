# ML Experiment Record

Use one record per experiment so that results can be interpreted and reproduced later.

## Question

What specific question are you trying to answer?

## Hypothesis

What outcome do you expect, and what is your reasoning?

## Data

- Dataset name and version:
- Number of examples:
- Features and target:
- Inclusion/exclusion criteria:
- Missing-value and duplicate handling:
- Train/validation/test split strategy:
- Potential leakage or sampling concerns:

## Method

- Baseline:
- Model or algorithm:
- Preprocessing:
- Hyperparameters:
- Random seed:
- Software and library versions:
- Hardware, if relevant:

## Evaluation

- Primary metric:
- Secondary metrics:
- Why these metrics fit the task:
- Exact evaluation data:
- Uncertainty estimate or repeated runs, if applicable:

## Results

| Run | Configuration | Primary metric | Secondary observations |
|---|---|---:|---|
| baseline |  |  |  |
| experiment-1 |  |  |  |
| experiment-2 |  |  |  |

## Interpretation

Separate observations from explanations. A metric difference is observed evidence; the reason for that difference may still be uncertain.

- What happened?
- Is the change large enough to matter?
- Which examples improved or worsened?
- Does the result hold across seeds or splits?
- Could leakage, noise, or an implementation issue explain it?

## Limitations

List what this experiment does not establish. Note dataset size, representativeness, missing comparisons, and any assumptions.

## Next experiment

Change one important factor and state what the result would teach you.
