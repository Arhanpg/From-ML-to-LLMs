"""MNIST digit classifier with a small feedforward neural network in NumPy.

TensorFlow/Keras is used only to download the MNIST dataset. The network,
activations, forward pass, gradients, and updates are implemented with NumPy.

Run from the repository root:
    python 05-deep-learning/mnist_numpy_nn.py

For a shorter smoke test, the script exposes epochs and batch size as CLI args:
    python 05-deep-learning/mnist_numpy_nn.py --epochs 1 --batch-size 2048

The default uses mini-batches for practical local training. The original
uploaded notebook uses full-batch gradient descent; see the accompanying
Markdown tutorial for a line-by-line explanation and the distinction.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    """Trainable weight matrices and bias vectors."""

    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels into shape (num_classes, examples) one-hot columns."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must contain integers")
    if num_classes < 2 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("labels must be in the range [0, num_classes)")

    encoded = np.zeros((labels.size, num_classes), dtype=np.float32)
    encoded[np.arange(labels.size), labels] = 1.0
    return encoded.T


def init_params(
    rng: np.random.Generator,
    input_size: int = 784,
    hidden_size: int = 64,
    num_classes: int = 10,
) -> Parameters:
    """Initialize weights with He-style scaling and biases with zeros."""
    if min(input_size, hidden_size, num_classes) <= 0:
        raise ValueError("all layer sizes must be positive")

    W1 = (
        rng.standard_normal((hidden_size, input_size)).astype(np.float32)
        * np.float32(np.sqrt(2.0 / input_size))
    )
    b1 = np.zeros((hidden_size, 1), dtype=np.float32)
    W2 = (
        rng.standard_normal((num_classes, hidden_size)).astype(np.float32)
        * np.float32(np.sqrt(2.0 / hidden_size))
    )
    b2 = np.zeros((num_classes, 1), dtype=np.float32)
    return Parameters(W1=W1, b1=b1, W2=W2, b2=b2)


def relu(z: np.ndarray) -> np.ndarray:
    """Rectified linear unit: zero for negative inputs, identity for positives."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """ReLU derivative/subgradient; this implementation chooses zero at z == 0."""
    return (z > 0.0).astype(z.dtype)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Convert class logits to probabilities, independently for each example column."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape (classes, examples)")
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=0, keepdims=True)


def forward_prop(
    params: Parameters, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Z1, A1, Z2, A2 for X shaped (input_features, examples)."""
    if X.ndim != 2 or X.shape[0] != params.W1.shape[1]:
        raise ValueError("X must be 2D and match the input layer size")
    Z1 = params.W1 @ X + params.b1
    A1 = relu(Z1)
    Z2 = params.W2 @ A1 + params.b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def cross_entropy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Mean categorical cross-entropy for one-hot targets."""
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must have the same 2D shape")
    if probabilities.shape[1] == 0:
        raise ValueError("cross-entropy requires at least one example")
    clipped = np.clip(probabilities, 1e-7, 1.0)
    return float(-np.sum(targets * np.log(clipped)) / targets.shape[1])


def backward_prop(
    Z1: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    params: Parameters,
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the mean-loss gradients dW1, db1, dW2, db2.

    This derivative is for softmax output with categorical cross-entropy.
    It uses the same matrix orientation as the uploaded learning notebook:
    examples are columns, and parameter gradients are averaged over examples.
    """
    if Y.ndim != 2 or Y.shape != A2.shape or Y.shape[1] == 0:
        raise ValueError("Y must match A2 and include at least one example")
    if X.shape[1] != Y.shape[1] or A1.shape[1] != Y.shape[1]:
        raise ValueError("X, A1, and Y must contain the same examples")

    m = Y.shape[1]
    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = (params.W2.T @ dZ2) * relu_derivative(Z1)
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return dW1, db1, dW2, db2


def update_params(
    params: Parameters,
    gradients: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    learning_rate: float,
) -> Parameters:
    """Take one gradient-descent step; backpropagation itself does not update weights."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    dW1, db1, dW2, db2 = gradients
    return Parameters(
        W1=params.W1 - learning_rate * dW1,
        b1=params.b1 - learning_rate * db1,
        W2=params.W2 - learning_rate * dW2,
        b2=params.b2 - learning_rate * db2,
    )


def accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Classification accuracy for probability and one-hot matrices."""
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must have the same 2D shape")
    predicted = np.argmax(probabilities, axis=0)
    actual = np.argmax(targets, axis=0)
    return float(np.mean(predicted == actual))


def evaluate(
    params: Parameters,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Evaluate loss and accuracy in chunks to avoid a large activation matrix."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    total_loss = 0.0
    correct = 0
    count = X.shape[1]
    if count == 0 or Y.shape[1] != count:
        raise ValueError("X and Y must contain the same non-empty number of examples")

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        targets = Y[:, start:stop]
        _, _, _, probs = forward_prop(params, X[:, start:stop])
        total_loss += cross_entropy(probs, targets) * (stop - start)
        correct += int(
            np.sum(np.argmax(probs, axis=0) == np.argmax(targets, axis=0))
        )
    return total_loss / count, correct / count


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST; TensorFlow is only the dataset loader, not the model framework."""
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is needed only for the MNIST dataset loader. "
            "Install dependencies from requirements-mnist.txt or run in Colab."
        ) from exc

    (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()
    x_train = x_train_raw.reshape(x_train_raw.shape[0], -1).T.astype(np.float32) / 255.0
    x_test = x_test_raw.reshape(x_test_raw.shape[0], -1).T.astype(np.float32) / 255.0
    y_train_encoded = one_hot(y_train)
    y_test_encoded = one_hot(y_test)
    return x_train, y_train_encoded, x_test, y_test_encoded


def train(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
    hidden_size: int = 64,
) -> Parameters:
    """Train with shuffled mini-batches and report post-epoch training metrics."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if X.shape[1] != Y.shape[1] or X.shape[1] == 0:
        raise ValueError("X and Y must contain the same non-empty number of examples")

    rng = np.random.default_rng(seed)
    params = init_params(
        rng, input_size=X.shape[0], hidden_size=hidden_size, num_classes=Y.shape[0]
    )

    for epoch in range(1, epochs + 1):
        order = rng.permutation(X.shape[1])
        for start in range(0, order.size, batch_size):
            indices = order[start : start + batch_size]
            x_batch = X[:, indices]
            y_batch = Y[:, indices]
            Z1, A1, _, A2 = forward_prop(params, x_batch)
            gradients = backward_prop(Z1, A1, A2, params, x_batch, y_batch)
            params = update_params(params, gradients, learning_rate)

        train_loss, train_accuracy = evaluate(params, X, Y, batch_size=1024)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy * 100:.2f}%"
        )

    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, Y_train, X_test, Y_test = load_mnist()
    print("Train input shape:", X_train.shape)
    print("Train target shape:", Y_train.shape)
    print("Test input shape: ", X_test.shape)
    print("Test target shape:", Y_test.shape)
    print(
        f"Architecture: {X_train.shape[0]} -> {args.hidden_size} -> {Y_train.shape[0]}"
    )
    print(
        f"Training: epochs={args.epochs}, batch_size={args.batch_size}, "
        f"learning_rate={args.learning_rate}, seed={args.seed}"
    )

    params = train(
        X_train,
        Y_train,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        hidden_size=args.hidden_size,
    )

    # The test set is evaluated after training, not used for gradient updates.
    test_loss, test_accuracy = evaluate(params, X_test, Y_test)
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
