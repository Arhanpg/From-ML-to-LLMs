"""A small context-window budgeting helper.

The values are user-supplied estimates; this script does not tokenize text or
query any model provider. Use the target model's official limits and tokenizer
for real requests.
"""


def calculate_budget(
    context_limit: int,
    system_tokens: int,
    user_tokens: int,
    retrieved_tokens: int,
    output_reserve: int,
    safety_margin: int = 0,
) -> dict[str, int | bool]:
    """Calculate whether estimated input and reserved output fit a context limit."""
    values = {
        "context_limit": context_limit,
        "system_tokens": system_tokens,
        "user_tokens": user_tokens,
        "retrieved_tokens": retrieved_tokens,
        "output_reserve": output_reserve,
        "safety_margin": safety_margin,
    }
    if any(value < 0 for value in values.values()):
        raise ValueError("token counts and context limit must be non-negative")
    if context_limit == 0:
        raise ValueError("context_limit must be positive")

    estimated_input = system_tokens + user_tokens + retrieved_tokens
    required_total = estimated_input + output_reserve + safety_margin
    available_for_retrieval = max(
        0, context_limit - system_tokens - user_tokens - output_reserve - safety_margin
    )

    return {
        "estimated_input_tokens": estimated_input,
        "reserved_output_tokens": output_reserve,
        "safety_margin_tokens": safety_margin,
        "required_total_tokens": required_total,
        "available_retrieval_tokens": available_for_retrieval,
        "fits": required_total <= context_limit,
    }


def main() -> None:
    budget = calculate_budget(
        context_limit=16_000,
        system_tokens=700,
        user_tokens=900,
        retrieved_tokens=10_000,
        output_reserve=2_000,
        safety_margin=500,
    )
    for key, value in budget.items():
        print(f"{key}: {value}")

    print(
        "\nThese are illustrative token counts, not measured counts. "
        "Use the intended model tokenizer and provider limits in production."
    )


if __name__ == "__main__":
    main()
