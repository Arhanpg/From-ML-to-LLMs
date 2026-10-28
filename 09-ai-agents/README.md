# 09 — AI Agents and Tool-Using Systems

An AI agent is often described as a system that uses a model to choose actions in pursuit of a goal, observes outcomes, and continues according to a control loop. The exact definition differs across frameworks, so focus on the behavior and system boundaries rather than the label.

## A minimal agent loop

1. Receive a goal and relevant state.
2. Ask the model to choose a supported action or final answer.
3. Validate the model's proposed action against a schema and policy.
4. Execute an allowed tool if needed.
5. Record the result and update state.
6. Stop when the goal is complete, a limit is reached, or human approval is required.

A workflow with predetermined steps can often be simpler and more reliable than a general autonomous agent. Start with ordinary code and add model-controlled actions only where flexible reasoning is genuinely useful.

## Tools

A tool should have a clear name, typed inputs, documented outputs, and bounded permissions. Examples include searching an authorized document collection, performing a calculation, or creating a draft.

Treat tool output as untrusted input. External documents and tool results can contain malformed data, irrelevant text, or instructions designed to influence a model.

## State and control flow

State may include the goal, previous tool calls, results, a task status, and a record of relevant evidence. Store only what is necessary, and define when it expires.

Useful control mechanisms include:
- maximum steps;
- deadlines and timeouts;
- retry limits with backoff;
- token and monetary budgets;
- schema validation;
- explicit stop conditions;
- human approval before consequential actions.

Do not let a model decide its own unlimited permissions or remove safety checks.

## Reliability and evaluation

Evaluate successful completion, correctness, tool selection, unnecessary actions, cost, latency, and unsafe behavior. Keep a set of known tasks and failure cases. Re-run them after changing prompts, tools, models, or control flow.

For consequential actions, use least privilege, preview changes, and require explicit authorization. Separate read-only tools from tools that modify external state.

## Prompt injection

Prompt injection occurs when untrusted content attempts to change how a model follows its instructions or uses tools. A retrieved document might say to reveal secrets or invoke an unrelated action.

Prompt wording alone is not a complete security boundary. Enforce permissions and validate action arguments in normal application code. Never give the model direct access to secrets or unrestricted execution simply because a prompt says to be careful.

## When to use an agent

Consider a more deterministic workflow when:
- the steps are known in advance;
- errors are expensive;
- actions have irreversible effects;
- output can be reliably computed by ordinary code.

Consider model-guided tool use when:
- the next step depends on varied natural-language inputs;
- available tools can be narrowly scoped;
- actions can be validated and failures safely handled;
- quality can be measured with representative tests.

## Exercises

1. Design a read-only agent that searches a small document collection.
2. Define its state, tool schema, and stopping conditions before implementing it.
3. Simulate a tool timeout and show how the workflow recovers.
4. Write an injected document instruction and verify it cannot expand tool permissions.
5. Compare the agent with a deterministic implementation on the same tasks.
