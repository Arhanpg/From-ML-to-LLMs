# LLM Application Reliability Checklist

Use this checklist when moving from a local prompt experiment toward an application.

## Inputs and context

- [ ] Input sizes and accepted formats are validated.
- [ ] The maximum context and output budgets are explicit.
- [ ] Retrieved or user-provided content is treated as untrusted data.
- [ ] Personal or confidential information is handled under a clear policy.
- [ ] Truncation is deliberate and observable.

## Prompt and output

- [ ] Instructions state the intended task and relevant constraints.
- [ ] Missing information is handled explicitly.
- [ ] Structured output is schema-validated when required.
- [ ] Unexpected refusals and unsafe completions are tested.
- [ ] Prompt changes are versioned and evaluated.

## Tools and retrieval

- [ ] Tool names and argument schemas are allowlisted.
- [ ] Tool permissions are enforced by application code.
- [ ] Consequential actions require appropriate authorization.
- [ ] Retrieval respects document-level access controls.
- [ ] Claims are checked against sources where grounding is required.
- [ ] Tool failures, timeouts, and malformed results have safe behavior.

## Evaluation and observability

- [ ] Representative test cases are versioned.
- [ ] Factual correctness and task completion are measured separately where relevant.
- [ ] Failure cases and regressions are reviewed.
- [ ] Latency, input/output tokens, errors, and approximate cost are monitored.
- [ ] Logs avoid unnecessary sensitive payloads.
- [ ] Evaluation includes adversarial and unanswerable inputs.

## Operations

- [ ] Requests have timeouts and bounded retries.
- [ ] Resource and monetary budgets are enforced.
- [ ] Model and prompt versions are recorded for diagnosis.
- [ ] A rollback or fallback path is defined.
- [ ] Human escalation is available for uncertain or high-impact cases.

This checklist is a starting point, not a certification. Requirements depend on domain, users, data sensitivity, and consequences of failure.
