# LLM Prompt Evaluation Template

Copy this template for a real prompt experiment. Record failures and not just favorable examples.

## Experiment identity

- Name:
- Date:
- Owner:
- Model and version:
- Prompt version:
- Intended use:
- Known constraints:

## Hypothesis

What behavior do you expect to improve, and why?

## Test set

Describe the source of the test cases. Include:
- typical requests;
- ambiguous or incomplete requests;
- edge cases;
- examples that should be refused or handled cautiously;
- examples where the right answer is to state that information is missing.

Do not use private or sensitive user data without appropriate authorization and protections.

## Controlled variables

Keep these fixed where possible:
- model version;
- system and developer instructions;
- inference parameters;
- available context;
- tools and tool permissions;
- evaluation method.

Record any factor that cannot be controlled.

## Metrics

Choose measures that match the task. Possible measures include:
- exact match for constrained outputs;
- task success rate;
- factual support by trusted references;
- schema validity;
- refusal correctness where relevant;
- tool-call correctness;
- latency and token usage;
- human review against a documented rubric.

A single aggregate score can hide important subgroups and failure types. Report the number and type of examples evaluated.

## Results table

| Case ID | Expected behavior | Observed behavior | Pass/fail | Failure category | Notes |
|---|---|---|---|---|---|
| case-001 |  |  |  |  |  |
| case-002 |  |  |  |  |  |
| case-003 |  |  |  |  |  |

## Failure analysis

For each important failure, record:
- input and relevant context;
- output or action;
- why it is incorrect or unsafe;
- whether the failure is reproducible;
- likely contributing factors;
- a proposed change;
- regression tests to add.

## Conclusion

- What improved?
- What regressed?
- What evidence supports the conclusion?
- What uncertainty remains?
- Should this version be deployed, tested further, or rejected?
