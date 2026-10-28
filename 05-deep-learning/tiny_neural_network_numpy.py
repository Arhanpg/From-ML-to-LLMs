"""Train a tiny two-layer neural network on the XOR pattern.

The goal is to make forward propagation and backpropagation concrete.
This educational program is intentionally small and is not a general framework.
Run from the repository root:
    python 05-deep-learning/tiny_neural_network_numpy.py
"""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Stable sigmoid function."""
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def binary_cross_entropy(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Binary cross-entropy with probability clipping for numerical safety."""
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(
        -np.mean(
            targets * np.log(clipped)
            + (1.0 - targets) * np.log(1.0 - clipped)
        )
    )


def main() -> None:
    features = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float
    )
    targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)

    rng = np.random.default_rng(seed=5)
    input_size = features.shape[1]
    hidden_size = 4
    output_size = 1

    weight_1 = rng.normal(0.0, 0.7, size=(input_size, hidden_size))
    bias_1 = np.zeros((1, hidden_size))
    weight_2 = rng.normal(0.0, 0.7, size=(hidden_size, output_size))
    bias_2 = np.zeros((1, output_size))

    learning_rate = 0.5
    iterations = 20_000
    count = features.shape[0]

    for step in range(iterations):
        # Forward pass: input -> hidden tanh -> output probability.
        hidden_pre_activation = features @ weight_1 + bias_1
        hidden_activation = np.tanh(hidden_pre_activation)
        output_logits = hidden_activation @ weight_2 + bias_2
        probabilities = sigmoid(output_logits)

        # For sigmoid + binary cross-entropy, dL/d(logits) is (p - y) / N.
        output_gradient = (probabilities - targets) / count
        weight_2_gradient = hidden_activation.T @ output_gradient
        bias_2_gradient = output_gradient.sum(axis=0, keepdims=True)

        hidden_gradient = (output_gradient @ weight_2.T) * (
            1.0 - hidden_activation**2
        )
        weight_1_gradient = features.T @ hidden_gradient
        bias_1_gradient = hidden_gradient.sum(axis=0, keepdims=True)

        weight_1 -= learning_rate * weight_1_gradient
        bias_1 -= learning_rate * bias_1_gradient
        weight_2 -= learning_rate * weight_2_gradient
        bias_2 -= learning_rate * bias_2_gradient

        if step in (0, 999, 4_999, iterations - 1):
            current_loss = binary_cross_entropy(
                targets,
                sigmoid(np.tanh(features @ weight_1 + bias_1) @ weight_2 + bias_2),
            )
            print(f"Step {step + 1:5d} | loss = {current_loss:.6f}")

    final_probabilities = sigmoid(
        np.tanh(features @ weight_1 + bias_1) @ weight_2 + bias_2
    )
    predicted = (final_probabilities >= 0.5).astype(int)

    print("\nInput -> probability -> prediction (target)")
    for row, probability, prediction, target in zip(
        features, final_probabilities[:, 0], predicted[:, 0], targets[:, 0]
    ):
        print(f"{row.astype(int)} -> {probability:.3f} -> {prediction} ({int(target)})")

    accuracy = float(np.mean(predicted == targets))
    print(f"Training accuracy on four XOR examples: {accuracy:.2f}")
    print("This tiny dataset demonstrates capacity, not generalization to real data.")


if __name__ == "__main__":
    main()
