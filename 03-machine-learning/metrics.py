"""Basic regression metrics implemented with NumPy."""

import numpy as np


def _validate_pair(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to arrays and validate them for these 1D metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have identical shapes")
    if actual.size == 0:
        raise ValueError("metrics require at least one observation")
    if actual.ndim != 1:
        raise ValueError("this educational implementation expects one-dimensional arrays")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("inputs must contain only finite values")

    return actual, predicted


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, expressed in target units."""
    actual, predicted = _validate_pair(actual, predicted)
    return float(np.mean(np.abs(actual - predicted)))


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean squared error, expressed in squared target units."""
    actual, predicted = _validate_pair(actual, predicted)
    return float(np.mean(np.square(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, expressed in target units."""
    return float(np.sqrt(mse(actual, predicted)))


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination relative to predicting the actual mean."""
    actual, predicted = _validate_pair(actual, predicted)
    total_sum_squares = float(np.sum((actual - actual.mean()) ** 2))
    if total_sum_squares == 0.0:
        raise ValueError("R-squared is undefined when actual values are constant")

    residual_sum_squares = float(np.sum((actual - predicted) ** 2))
    return 1.0 - residual_sum_squares / total_sum_squares


def main() -> None:
    actual = np.array([3.0, -0.5, 2.0, 7.0])
    predicted = np.array([2.5, 0.0, 2.0, 8.0])

    print(f"MAE: {mae(actual, predicted):.3f}")
    print(f"MSE: {mse(actual, predicted):.3f}")
    print(f"RMSE: {rmse(actual, predicted):.3f}")
    print(f"R-squared: {r_squared(actual, predicted):.3f}")


if __name__ == "__main__":
    main()
