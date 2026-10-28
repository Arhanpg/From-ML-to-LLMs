"""Linear regression trained with NumPy gradient descent.

Synthetic data keeps the example reproducible and self-contained.
Run from the repository root:
    python 03-machine-learning/linear_regression_from_scratch.py
"""

import numpy as np


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return mean squared prediction error after validating array shapes."""
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual.size == 0:
        raise ValueError("metrics require at least one observation")
    return float(np.mean((actual - predicted) ** 2))


def train_gradient_descent(
    design_matrix: np.ndarray,
    target: np.ndarray,
    learning_rate: float = 0.1,
    iterations: int = 2_000,
) -> tuple[np.ndarray, list[float]]:
    """Fit parameters by minimizing MSE with batch gradient descent."""
    if design_matrix.ndim != 2:
        raise ValueError("design_matrix must be two-dimensional")
    if target.ndim != 1 or design_matrix.shape[0] != target.shape[0]:
        raise ValueError("target must be one-dimensional and match the row count")
    if target.size == 0:
        raise ValueError("training data must not be empty")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    row_count, _ = design_matrix.shape
    parameters = np.zeros(design_matrix.shape[1], dtype=float)
    losses: list[float] = []

    for _ in range(iterations):
        predictions = design_matrix @ parameters
        errors = predictions - target
        gradient = (2.0 / row_count) * (design_matrix.T @ errors)
        parameters -= learning_rate * gradient
        losses.append(mean_squared_error(target, design_matrix @ parameters))

    return parameters, losses


def main() -> None:
    rng = np.random.default_rng(seed=7)
    feature = rng.uniform(0.0, 10.0, size=160)
    noise = rng.normal(loc=0.0, scale=2.0, size=feature.size)
    target = 4.2 * feature + 7.5 + noise

    permutation = rng.permutation(feature.size)
    split_at = int(0.8 * feature.size)
    train_index = permutation[:split_at]
    test_index = permutation[split_at:]

    x_train = feature[train_index]
    x_test = feature[test_index]
    y_train = target[train_index]
    y_test = target[test_index]

    # Fit preprocessing statistics only on the training subset.
    mean = float(x_train.mean())
    std = float(x_train.std())
    if std == 0.0:
        raise ValueError("training feature has zero variance")

    x_train_scaled = (x_train - mean) / std
    x_test_scaled = (x_test - mean) / std
    design_train = np.column_stack([np.ones(x_train.size), x_train_scaled])
    design_test = np.column_stack([np.ones(x_test.size), x_test_scaled])

    parameters, losses = train_gradient_descent(design_train, y_train)
    train_prediction = design_train @ parameters
    test_prediction = design_test @ parameters

    # Convert the learned parameters back to the feature's original units.
    slope = float(parameters[1] / std)
    intercept = float(parameters[0] - parameters[1] * mean / std)

    print("Learned model")
    print(f"  prediction = {slope:.3f} * feature + {intercept:.3f}")
    print(f"  final training MSE: {losses[-1]:.3f}")
    print(f"  train MSE: {mean_squared_error(y_train, train_prediction):.3f}")
    print(f"  test MSE:  {mean_squared_error(y_test, test_prediction):.3f}")
    print(
        "Underlying synthetic relationship: target = 4.2 * feature + 7.5 + noise"
    )
    print("Learned coefficients differ slightly because the target includes noise.")


if __name__ == "__main__":
    main()
