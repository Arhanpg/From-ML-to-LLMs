"""Create a basic plot using synthetic data.

Run from the repository root:
    python 02-python-for-ml/matplotlib_basics.py

The chart is saved to artifacts/linear_relationship.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    rng = np.random.default_rng(seed=12)
    x = np.linspace(0.0, 10.0, num=60)
    y = 2.0 * x + 3.0 + rng.normal(loc=0.0, scale=2.0, size=x.size)

    output_path = Path("artifacts") / "linear_relationship.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, label="Synthetic observations")
    ax.plot(x, 2.0 * x + 3.0, label="Underlying relationship")
    ax.set_title("A simple relationship with noise")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved chart to: {output_path}")
    print("The line shows the underlying relationship; points include random noise.")


if __name__ == "__main__":
    main()
