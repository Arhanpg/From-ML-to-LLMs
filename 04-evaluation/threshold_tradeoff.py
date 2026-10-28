"""Show how classification metrics change with a decision threshold."""

import numpy as np


def metrics_at_threshold(
    actual: np.ndarray, scores: np.ndarray, threshold: float
) -> tuple[float, float, float, int, int, int, int]:
    """Return accuracy, precision, recall, and TN/FP/FN/TP for one threshold."""
    if actual.ndim != 1 or scores.ndim != 1 or actual.shape != scores.shape:
        raise ValueError("actual and scores must be matching 1D arrays")
    if actual.size == 0:
        raise ValueError("evaluation data must not be empty")
    if not np.all(np.isin(actual, [0, 1])):
        raise ValueError("actual labels must contain only 0 and 1")
    if not np.all(np.isfinite(scores)) or not np.all((scores >= 0) & (scores <= 1)):
        raise ValueError("scores must be finite values between 0 and 1")

    predicted = (scores >= threshold).astype(int)
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))

    accuracy = (tn + tp) / actual.size
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return accuracy, precision, recall, tn, fp, fn, tp


def main() -> None:
    actual = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.05, 0.12, 0.28, 0.47, 0.31, 0.51, 0.66, 0.79, 0.93])

    print("Threshold | Accuracy | Precision | Recall | TN FP FN TP")
    for threshold in (0.25, 0.50, 0.75):
        acc, prec, rec, tn, fp, fn, tp = metrics_at_threshold(
            actual, scores, threshold
        )
        print(
            f"{threshold:9.2f} | {acc:8.3f} | {prec:9.3f} | {rec:6.3f} "
            f"| {tn:2d} {fp:2d} {fn:2d} {tp:2d}"
        )

    print("\nA lower threshold often finds more positives but can raise false positives.")
    print("Select a threshold using validation data and the costs of each error.")


if __name__ == "__main__":
    main()
