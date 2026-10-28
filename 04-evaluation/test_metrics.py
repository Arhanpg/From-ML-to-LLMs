"""Unit tests for regression metrics, runnable without pytest.

Run from the repository root:
    python -m unittest discover -s 04-evaluation -p 'test_*.py'
"""

import unittest

import numpy as np

from pathlib import Path
import sys

# Permit direct script-based tests without packaging the repository as a module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "03-machine-learning"))

from metrics import mae, mse, rmse, r_squared  # noqa: E402


class RegressionMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.actual = np.array([1.0, 2.0, 3.0])
        self.predicted = np.array([1.0, 3.0, 2.0])

    def test_mae(self) -> None:
        self.assertAlmostEqual(mae(self.actual, self.predicted), 2.0 / 3.0)

    def test_mse_and_rmse(self) -> None:
        self.assertAlmostEqual(mse(self.actual, self.predicted), 2.0 / 3.0)
        self.assertAlmostEqual(rmse(self.actual, self.predicted), np.sqrt(2.0 / 3.0))

    def test_perfect_prediction_has_r_squared_one(self) -> None:
        self.assertAlmostEqual(r_squared(self.actual, self.actual), 1.0)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mae(np.array([1.0]), np.array([1.0, 2.0]))

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mse(np.array([]), np.array([]))

    def test_constant_target_r_squared_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            r_squared(np.array([2.0, 2.0]), np.array([2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
