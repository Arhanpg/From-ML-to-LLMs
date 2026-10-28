"""Unit tests for binary classification metrics.

Run from the repository root:
    python -m unittest discover -s 04-evaluation -p 'test_*.py'
"""

import unittest

import numpy as np

from classification_metrics import accuracy, confusion_counts, f1_score, precision, recall


class ClassificationMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = np.array([0, 0, 0, 1, 1, 1, 1])
        self.predicted = np.array([0, 0, 1, 1, 0, 1, 1])

    def test_confusion_counts(self) -> None:
        self.assertEqual(confusion_counts(self.actual, self.predicted), (2, 1, 1, 3))

    def test_accuracy(self) -> None:
        self.assertAlmostEqual(accuracy(self.actual, self.predicted), 5.0 / 7.0)

    def test_precision_recall_and_f1(self) -> None:
        self.assertAlmostEqual(precision(self.actual, self.predicted), 0.75)
        self.assertAlmostEqual(recall(self.actual, self.predicted), 0.75)
        self.assertAlmostEqual(f1_score(self.actual, self.predicted), 0.75)

    def test_zero_positive_predictions_return_zero_precision_and_f1(self) -> None:
        predicted = np.zeros_like(self.actual)
        self.assertEqual(precision(self.actual, predicted), 0.0)
        self.assertEqual(f1_score(self.actual, predicted), 0.0)

    def test_invalid_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confusion_counts(np.array([0, 2]), np.array([0, 1]))

    def test_mismatched_shapes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            accuracy(np.array([0]), np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
