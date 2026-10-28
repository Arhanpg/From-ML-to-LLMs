"""Demonstrate standardization fitted only on training data."""

import numpy as np


def fit_standardizer(training_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate per-column mean and standard deviation from training rows."""
    if training_features.ndim != 2 or training_features.shape[0] == 0:
        raise ValueError("training_features must be a non-empty 2D matrix")
    if not np.all(np.isfinite(training_features)):
        raise ValueError("training features must be finite")

    mean = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    # A constant feature maps to zero rather than dividing by zero.
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def transform(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply previously fitted statistics without refitting them."""
    if features.ndim != 2 or features.shape[1] != mean.size:
        raise ValueError("feature count must match fitted statistics")
    if scale.shape != mean.shape or np.any(scale <= 0.0):
        raise ValueError("scale must be positive and match mean")
    return (features - mean) / scale


def main() -> None:
    train = np.array([[10.0, 100.0], [12.0, 150.0], [14.0, 200.0]])
    test = np.array([[20.0, 500.0]])

    train_mean, train_scale = fit_standardizer(train)
    transformed_train = transform(train, train_mean, train_scale)
    transformed_test = transform(test, train_mean, train_scale)

    print("Training means:", train_mean)
    print("Training scales:", train_scale)
    print("Transformed training data:")
    print(transformed_train)
    print("Transformed test data using training statistics:")
    print(transformed_test)

    print("\nThe test point can lie far from zero; do not refit scaling on test data.")
    print("An out-of-range standardized value may be a useful signal, not a bug.")


if __name__ == "__main__":
    main()
