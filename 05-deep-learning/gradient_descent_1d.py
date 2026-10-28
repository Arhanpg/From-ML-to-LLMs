"""Use gradient descent to minimize a one-dimensional quadratic loss."""

def loss(parameter: float) -> float:
    return (parameter - 3.0) ** 2


def gradient(parameter: float) -> float:
    return 2.0 * (parameter - 3.0)


def optimize(
    initial_parameter: float, learning_rate: float, steps: int
) -> list[tuple[int, float, float]]:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if steps < 0:
        raise ValueError("steps must be non-negative")

    parameter = initial_parameter
    history = [(0, parameter, loss(parameter))]
    for step in range(1, steps + 1):
        parameter -= learning_rate * gradient(parameter)
        history.append((step, parameter, loss(parameter)))
    return history


def main() -> None:
    for learning_rate in (0.1, 0.5, 1.1):
        history = optimize(initial_parameter=-4.0, learning_rate=learning_rate, steps=8)
        print(f"Learning rate: {learning_rate}")
        for step, parameter, current_loss in history:
            print(
                f"  step={step:2d} parameter={parameter:9.4f} "
                f"loss={current_loss:12.6f}"
            )
        print()

    print("The minimum is at parameter = 3.")
    print("A learning rate above 1 overshoots and grows unstable for this quadratic.")


if __name__ == "__main__":
    main()
