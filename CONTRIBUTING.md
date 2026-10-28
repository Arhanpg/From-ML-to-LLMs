# Contributing to this learning repository

New material should make a concept easier to understand, test, or reproduce.

## Suggested structure for a topic

1. Goal: what question does this topic answer?
2. Intuition: explain the idea in plain language.
3. Key terms: define unfamiliar words and symbols.
4. How it works: show the main steps or equations.
5. Code: include a small runnable example.
6. Expected output: explain what the learner should observe.
7. Experiment: change one thing and describe the result.
8. Limitations: state assumptions and edge cases.
9. Exercises: ask questions that require application.
10. References: link to useful documentation or papers.

Keep the explanation proportional to the difficulty. Not every short example needs all ten sections.

## Code guidelines

- Prefer small, self-contained programs.
- Use descriptive names and type hints where they improve readability.
- Fix random seeds in demonstrations that rely on randomness.
- Avoid private local paths and unnecessary network downloads.
- Check important input assumptions, such as shapes and empty inputs.
- Explain mathematical operations rather than hiding everything behind a library.
- Do not commit API keys, access tokens, passwords, private data, or large datasets.

## Experiment guidelines

A useful experiment states the question, baseline, variable changed, metric, result, and limitations. Change one factor at a time when learning the effect of a parameter.

A single experiment is evidence for that setup, not proof that a result generalizes everywhere.

## Before submitting a change

- Check whether a similar lesson already exists.
- Run code in a clean environment when possible.
- Check relative paths, output, and assumptions.
- Use a commit message that describes the actual change.
- Prefer meaningful educational improvements over empty or duplicated material.
