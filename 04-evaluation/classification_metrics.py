"""Common binary-classification metrics implemented with NumPy."""

import numpy as np


def confusion_counts(
    actual: np.ndarray, predicted: np.ndarray
) -> tuple[int, int, int, int]:
    """Return true negatives, false positives, false negatives, true positives."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("inputs must be one-dimensional")
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("inputs must have matching, non-empty shapes")
    if not np.all(np.isin(actual, [0, 1])) or not np.all(np.isin(predicted, [0, 1])):
        raise ValueError("labels must contain only 0 and 1")

    actual = actual.astype(int)
    predicted = predicted.astype(int)
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    return tn, fp, fn, tp


def _ratio(numerator: int, denominator: int) -> float:
    """Return zero when a metric's denominator is zero."""
    return float(numerator / denominator) if denominator else 0.0


def accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_counts(actual, predicted)
    return _ratio(tn + tp, tn + fp + fn + tp)


def precision(actual: np.ndarray, predicted: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_counts(actual, predicted)
    return _ratio(tp, tp + fp)


def recall(actual: np.ndarray, predicted: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_counts(actual, predicted)
    return _ratio(tp, tp + fn)


def f1_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    p = precision(actual, predicted)
    r = recall(actual, predicted)
    return _ratio(int(round(2.0 * p * r * 1_000_000)), int(round((p + r) * 1_000_000)))


def main() -> None:
    actual = np.array([0, 0, 0, 1, 1, 1, 1])
    predicted = np.array([0, 0, 1, 1, 0, 1, 1])
    tn, fp, fn, tp = confusion_counts(actual, predicted)

    print("Confusion counts")
    print(f"  true negatives:  {tn}")
    print(f"  false positives: {fp}")
    print(f"  false negatives: {fn}")
    print(f"  true positives:  {tp}")
    print(f"Accuracy:  {accuracy(actual, predicted):.3f}")
    print(f"Precision: {precision(actual, predicted):.3f}")
    print(f"Recall:    {recall(actual, predicted):.3f}")
    print(f"F1:        {f1_score(actual, predicted):.3f}")


if __name__ == "__main__":
    main()
