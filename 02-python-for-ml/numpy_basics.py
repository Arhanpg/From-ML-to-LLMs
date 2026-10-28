"""Small examples of NumPy fundamentals.

Run from the repository root:
    python 02-python-for-ml/numpy_basics.py
"""

import numpy as np


def main() -> None:
    features = np.array(
        [
            [1.0, 20.0, 0.0],
            [2.0, 25.0, 1.0],
            [3.0, 30.0, 0.0],
        ]
    )
    print("Feature matrix:")
    print(features)
    print("Shape:", features.shape)
    print("Number of dimensions:", features.ndim)

    # A boolean mask selects every row whose second feature exceeds 24.
    ages = features[:, 1]
    selected_rows = features[ages > 24]
    print("\nRows selected by a boolean mask:")
    print(selected_rows)

    # Vectorization applies an operation to all values without a Python loop.
    prices = np.array([100.0, 150.0, 80.0])
    discounted_prices = prices * (1.0 - 0.10)
    print("\nDiscounted prices:", discounted_prices)

    # Broadcasting subtracts one value per column from every row.
    column_means = features.mean(axis=0, keepdims=True)
    centered = features - column_means
    print("\nColumn means:", column_means)
    print("Centered matrix:")
    print(centered)

    print("\nSum by column:", features.sum(axis=0))
    print("Sum by row:", features.sum(axis=1))

    # A local random generator avoids changing NumPy's global random state.
    rng = np.random.default_rng(seed=42)
    print("\nFive reproducible random values:", rng.normal(size=5))


if __name__ == "__main__":
    main()
