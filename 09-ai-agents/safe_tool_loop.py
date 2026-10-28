"""Demonstrate a bounded tool loop with validation and an explicit allowlist.

No LLM or external service is called. The proposed actions are sample data.
Run from the repository root:
    python 09-ai-agents/safe_tool_loop.py
"""

from typing import Any


def add_numbers(arguments: dict[str, Any]) -> dict[str, float]:
    """Add two finite numeric inputs after strict validation."""
    if set(arguments) != {"left", "right"}:
        raise ValueError("add requires exactly left and right arguments")

    left = arguments["left"]
    right = arguments["right"]
    for name, value in (("left", left), ("right", right)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number, not a boolean or string")
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{name} must be finite")

    return {"result": float(left + right)}


TOOLS = {
    "add": add_numbers,
}


def execute_tool(action: dict[str, Any]) -> dict[str, Any]:
    """Execute only a registered tool and return a structured result."""
    name = action.get("tool")
    arguments = action.get("arguments", {})

    if not isinstance(name, str) or name not in TOOLS:
        return {"ok": False, "error": "Tool is not allowlisted"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "Tool arguments must be an object"}

    try:
        return {"ok": True, "tool": name, "output": TOOLS[name](arguments)}
    except (TypeError, ValueError) as exc:
        return {"ok": False, "tool": name, "error": str(exc)}


def run_bounded_loop(actions: list[dict[str, Any]], max_steps: int = 3) -> None:
    """Process at most max_steps proposed actions and print the outcomes."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    for step, action in enumerate(actions[:max_steps], start=1):
        result = execute_tool(action)
        print(f"Step {step}: {result}")


def main() -> None:
    proposed_actions = [
        {"tool": "add", "arguments": {"left": 12, "right": 30}},
        {"tool": "run_shell", "arguments": {"command": "not allowed"}},
        {"tool": "add", "arguments": {"left": "12", "right": 30}},
        {"tool": "add", "arguments": {"left": 1, "right": 2}},
    ]

    run_bounded_loop(proposed_actions, max_steps=3)
    print("\nThe fourth action is not run because the step limit is three.")
    print("A production system also needs authentication, audit logs, timeouts, and policy checks.")


if __name__ == "__main__":
    main()
