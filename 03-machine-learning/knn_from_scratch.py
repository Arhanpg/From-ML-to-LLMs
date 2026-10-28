"""A transparent k-nearest neighbors classifier using NumPy.

KNN is a lazy learner: it stores training examples and predicts from nearby
examples instead of fitting a parameterized training objective.

Run from the repository root:
    python 03-machine-learning/knn_from_scratch.py
"""

import numpy as np


def predict_one(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query: np.ndarray,
    k: int = 3,
) -> int | float | str:
    """Predict one label using Euclidean distance and majority voting."""
    if train_features.ndim != 2:
        raise ValueError("train_features must be a 2D matrix")
    if train_labels.ndim != 1 or train_labels.size != train_features.shape[0]:
        raise ValueError("train_labels must match the number of training rows")
    if query.ndim != 1 or query.size != train_features.shape[1]:
        raise ValueError("query must be a vector matching the feature count")
    if train_features.shape[0] == 0:
        raise ValueError("training data must not be empty")
    if k < 1 or k > train_features.shape[0]:
        raise ValueError("k must be between 1 and the number of training rows")

    distances_squared = np.sum((train_features - query) ** 2, axis=1)
    nearest_indices = np.argpartition(distances_squared, k - 1)[:k]
    nearest_labels = train_labels[nearest_indices]

    labels, counts = np.unique(nearest_labels, return_counts=True)
    # np.unique returns sorted labels, making ties deterministic for comparable labels.
    return labels[int(np.argmax(counts))].item()


def predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    queries: np.ndarray,
    k: int = 3,
) -> np.ndarray:
    """Predict labels for multiple query rows."""
    if queries.ndim != 2 or queries.shape[1] != train_features.shape[1]:
        raise ValueError("queries must be a 2D matrix with matching feature count")
    return np.asarray(
        [predict_one(train_features, train_labels, row, k) for row in queries]
    )


def main() -> None:
    # Two compact clusters make the distance-based idea easy to see.
    x_train = np.array(
        [
            [0.0, 0.1],
            [0.2, 0.0],
            [0.1, 0.3],
            [2.0, 2.1],
            [2.2, 2.0],
            [1.9, 2.2],
        ]
    )
    y_train = np.array(["red", "red", "red", "blue", "blue", "blue"])
    x_test = np.array([[0.15, 0.12], [2.1, 2.05]])

    predictions = predict(x_train, y_train, x_test, k=3)
    for point, label in zip(x_test, predictions):
        print(f"Point {point} -> predicted class: {label}")

    print("\nKNN is distance-based: feature scaling can strongly affect results.")
    print("Choose k using validation data rather than repeatedly checking the test set.")


if __name__ == "__main__":
    main()
