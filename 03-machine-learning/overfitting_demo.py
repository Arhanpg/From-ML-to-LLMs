"""Compare polynomial fits to illustrate underfitting and overfitting.

The experiment is deliberately small and synthetic. Test-set results can vary
with the random seed and should not be generalized beyond this demonstration.
Run from the repository root:
    python 03-machine-learning/overfitting_demo.py
"""

import numpy as np


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def main() -> None:
    rng = np.random.default_rng(seed=24)
    x = np.sort(rng.uniform(-3.0, 3.0, size=45))
    y = np.sin(x) + rng.normal(loc=0.0, scale=0.18, size=x.size)

    order = rng.permutation(x.size)
    split = int(0.7 * x.size)
    train_index, test_index = order[:split], order[split:]
    x_train, y_train = x[train_index], y[train_index]
    x_test, y_test = x[test_index], y[test_index]

    print("Degree | Train MSE | Test MSE")
    print("-----------------------------")
    for degree in (1, 3, 8, 12):
        coefficients = np.polyfit(x_train, y_train, deg=degree)
        train_prediction = np.polyval(coefficients, x_train)
        test_prediction = np.polyval(coefficients, x_test)
        print(
            f"{degree:6d} | {mean_squared_error(y_train, train_prediction):9.4f} "
            f"| {mean_squared_error(y_test, test_prediction):8.4f}"
        )

    print("\nInterpretation:")
    print("- Degree 1 may be too simple to capture a curved relationship.")
    print("- Higher degrees can fit training data closely.")
    print("- A small training error does not guarantee a small test error.")
    print("Use a validation strategy to choose complexity; keep the test set for evaluation.")


if __name__ == "__main__":
    main()
