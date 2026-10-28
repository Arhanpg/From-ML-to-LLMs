"""Unit tests for the NumPy MNIST network.

The tests do not download MNIST or require TensorFlow.
Run from the repository root:
    python -m unittest discover -s 05-deep-learning -p 'test_mnist*.py'
"""

import unittest

import numpy as np

from mnist_numpy_nn import (
    Parameters,
    backward_prop,
    cross_entropy,
    forward_prop,
    init_params,
    one_hot,
    softmax,
)


class MNISTNetworkTests(unittest.TestCase):
    def test_one_hot_encoding_has_expected_orientation(self) -> None:
        encoded = one_hot(np.array([2, 0, 1]), num_classes=3)
        self.assertEqual(encoded.shape, (3, 3))
        np.testing.assert_array_equal(np.argmax(encoded, axis=0), [2, 0, 1])
        np.testing.assert_array_equal(encoded.sum(axis=0), [1, 1, 1])

    def test_softmax_is_finite_and_each_column_sums_to_one(self) -> None:
        logits = np.array([[1000.0, -1000.0], [1001.0, 0.0], [999.0, 1000.0]])
        probabilities = softmax(logits)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities.sum(axis=0), [1.0, 1.0], atol=1e-12)

    def test_forward_shapes_and_probability_normalization(self) -> None:
        params = init_params(np.random.default_rng(7), input_size=4, hidden_size=3, num_classes=2)
        features = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        Z1, A1, Z2, A2 = forward_prop(params, features)
        self.assertEqual(Z1.shape, (3, 2))
        self.assertEqual(A1.shape, (3, 2))
        self.assertEqual(Z2.shape, (2, 2))
        self.assertEqual(A2.shape, (2, 2))
        np.testing.assert_allclose(A2.sum(axis=0), [1.0, 1.0], atol=1e-6)

    def test_backprop_gradients_match_finite_differences(self) -> None:
        # Small deterministic network chosen so selected ReLU pre-activations
        # are away from zero, where finite-difference comparisons are unstable.
        params = Parameters(
            W1=np.array([[0.2, -0.1], [0.4, 0.3]], dtype=float),
            b1=np.array([[0.1], [-0.05]], dtype=float),
            W2=np.array([[0.3, -0.2], [-0.4, 0.5]], dtype=float),
            b2=np.array([[0.05], [-0.07]], dtype=float),
        )
        X = np.array([[0.2, -0.4, 0.7], [0.8, 0.5, -0.3]], dtype=float)
        Y = one_hot(np.array([0, 1, 0]), num_classes=2).astype(float)

        Z1, A1, _, A2 = forward_prop(params, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, params, X, Y)

        def loss() -> float:
            return cross_entropy(forward_prop(params, X)[3], Y)

        def finite_difference(array_name: str, index: tuple[int, ...], epsilon: float = 1e-5) -> float:
            array = getattr(params, array_name)
            original = float(array[index])
            array[index] = original + epsilon
            plus = loss()
            array[index] = original - epsilon
            minus = loss()
            array[index] = original
            return (plus - minus) / (2.0 * epsilon)

        checks = [
            ("W1", (0, 0), dW1[0, 0]),
            ("b1", (0, 0), db1[0, 0]),
            ("W2", (0, 0), dW2[0, 0]),
            ("b2", (1, 0), db2[1, 0]),
        ]
        for array_name, index, analytical in checks:
            numerical = finite_difference(array_name, index)
            self.assertAlmostEqual(
                float(analytical),
                numerical,
                delta=2e-5,
                msg=f"Gradient mismatch for {array_name}{index}",
            )

    def test_cross_entropy_is_finite(self) -> None:
        probabilities = np.array([[0.8, 0.2], [0.2, 0.8]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertGreaterEqual(cross_entropy(probabilities, targets), 0.0)
        self.assertTrue(np.isfinite(cross_entropy(probabilities, targets)))


if __name__ == "__main__":
    unittest.main()
