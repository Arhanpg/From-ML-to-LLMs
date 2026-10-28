"""Binary logistic regression trained with NumPy gradient descent.

The program generates a small synthetic dataset and uses a held-out test set.
Run from the repository root:
    python 03-machine-learning/logistic_regression_from_scratch.py
"""

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for finite real-valued logits."""
    result = np.empty_like(logits, dtype=float)
    nonnegative = logits >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_values = np.exp(logits[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def binary_cross_entropy(
    targets: np.ndarray, probabilities: np.ndarray
) -> float:
    """Compute binary cross-entropy after clipping probabilities for log safety."""
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape")
    if targets.size == 0:
        raise ValueError("loss requires at least one observation")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("targets must contain only 0 and 1")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    )


def train(
    design: np.ndarray,
    targets: np.ndarray,
    learning_rate: float = 0.1,
    iterations: int = 3_000,
) -> tuple[np.ndarray, list[float]]:
    """Fit logistic-regression weights with batch gradient descent."""
    if design.ndim != 2 or targets.ndim != 1:
        raise ValueError("design must be 2D and targets must be 1D")
    if design.shape[0] != targets.size or targets.size == 0:
        raise ValueError("design rows must match a non-empty target vector")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise ValueError("targets must contain only 0 and 1")
    if learning_rate <= 0 or iterations <= 0:
        raise ValueError("learning_rate and iterations must be positive")

    weights = np.zeros(design.shape[1], dtype=float)
    losses: list[float] = []
    for _ in range(iterations):
        probabilities = sigmoid(design @ weights)
        gradient = design.T @ (probabilities - targets) / targets.size
        weights -= learning_rate * gradient
        losses.append(binary_cross_entropy(targets, sigmoid(design @ weights)))
    return weights, losses


def main() -> None:
    rng = np.random.default_rng(seed=19)
    count = 500
    features = rng.normal(size=(count, 2))
    signal = 1.8 * features[:, 0] - 1.2 * features[:, 1] + 0.3
    probabilities = sigmoid(signal)
    labels = (rng.random(count) < probabilities).astype(float)

    indices = rng.permutation(count)
    split = int(0.8 * count)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # Standardize based on training data only.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    design_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    design_test = np.column_stack([np.ones(x_test.shape[0]), x_test])

    weights, losses = train(design_train, y_train)
    test_probabilities = sigmoid(design_test @ weights)
    predicted_labels = (test_probabilities >= 0.5).astype(float)
    accuracy = float(np.mean(predicted_labels == y_test))

    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final training loss: {losses[-1]:.4f}")
    print(f"Test accuracy: {accuracy:.3f}")
    print("Accuracy is only one metric; inspect precision, recall, and class balance too.")


if __name__ == "__main__":
    main()
