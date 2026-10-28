"""A tiny scaled dot-product attention calculation using NumPy.

This is an illustration of the operation, not a trained transformer.
Run from the repository root:
    python 06-transformers/self_attention_numpy.py
"""

import numpy as np


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return attention output and weights for rank-2 sequence matrices.

    query: (query_length, depth)
    key:   (key_length, depth)
    value: (key_length, value_depth)
    mask: optional boolean matrix broadcastable to (query_length, key_length).
          True means the position is allowed; False means it is masked.
    """
    if query.ndim != 2 or key.ndim != 2 or value.ndim != 2:
        raise ValueError("query, key, and value must be two-dimensional")
    if query.shape[1] != key.shape[1]:
        raise ValueError("query and key depths must match")
    if key.shape[0] != value.shape[0]:
        raise ValueError("key and value sequence lengths must match")
    if query.shape[1] == 0 or key.shape[0] == 0:
        raise ValueError("sequence lengths and depth must be non-zero")

    scores = query @ key.T / np.sqrt(query.shape[1])

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        try:
            allowed = np.broadcast_to(mask, scores.shape)
        except ValueError as exc:
            raise ValueError("mask must be broadcastable to the score shape") from exc

        if not np.all(allowed.any(axis=1)):
            raise ValueError("each query row must have at least one allowed key")
        scores = np.where(allowed, scores, -np.inf)

    weights = softmax(scores, axis=1)
    output = weights @ value
    return output, weights


def main() -> None:
    # Three token-like positions, each represented by a 2D vector.
    query = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    key = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    value = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

    output, weights = scaled_dot_product_attention(query, key, value)
    print("Attention weights (each row sums to 1):")
    print(np.round(weights, 3))
    print("Row sums:", np.round(weights.sum(axis=1), 6))
    print("\nWeighted value outputs:")
    print(np.round(output, 3))

    # A causal-style mask lets each position attend only to itself and earlier positions.
    causal_mask = np.tril(np.ones((3, 3), dtype=bool))
    causal_output, causal_weights = scaled_dot_product_attention(
        query, key, value, mask=causal_mask
    )
    print("\nCausal attention weights:")
    print(np.round(causal_weights, 3))
    print("Causal outputs:")
    print(np.round(causal_output, 3))


if __name__ == "__main__":
    main()
