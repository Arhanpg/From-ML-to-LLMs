# 07 — LLM Engineering

An LLM application is more than a prompt sent to a model. Useful systems combine inputs, inference settings, context management, tool or retrieval integration, evaluation, and safeguards.

## Next-token prediction

Many widely used generative language models are trained to predict the next token given preceding tokens. During generation, the model produces scores over possible next tokens, selects or samples one, appends it to the context, and repeats.

The exact training and inference procedures vary by model family. An LLM is not a database that guarantees it has memorized every fact correctly, and fluent output is not proof of factual correctness.

## Prompting

A practical prompt often makes these items explicit:

- the task and desired outcome;
- relevant context and source material;
- constraints and prohibited assumptions;
- the expected output format;
- what to do when information is missing.

Examples help define a pattern, but do not eliminate ambiguity or guarantee correctness. Test prompts on varied cases, including adversarial or underspecified inputs.

## Sampling controls

- **Greedy decoding:** selects the highest-scoring next token at each step.
- **Temperature:** rescales logits before sampling; lower values usually make sampling more concentrated, while higher values generally increase randomness.
- **Top-k:** restricts sampling to a set of the k highest-ranked tokens.
- **Top-p:** restricts sampling to a smallest high-probability set whose cumulative probability reaches a threshold.

Parameters interact, and implementation details differ between runtimes. Compare them on the same representative evaluation set.

## Context windows and cost

The context window limits how many tokens can be considered together. Input instructions, retrieved documents, prior conversation, and generated output all consume context according to the model/runtime's accounting rules.

More context is not automatically better. Noisy or irrelevant documents can distract the model, increase cost, and leave less room for the answer. Track latency, token usage, error rate, and task quality together.

## Fine-tuning and parameter-efficient methods

- **Pretraining** learns broad sequence patterns from large corpora.
- **Supervised fine-tuning** trains on examples of desired inputs and outputs.
- **Preference optimization** attempts to align behavior with preference signals using a particular training method.
- **LoRA** learns low-rank adaptation matrices while keeping much of the original weight set fixed.
- **QLoRA** combines quantized base-model weights with trainable low-rank adapters in a memory-efficient fine-tuning setup.

Fine-tuning is not always the best answer to a knowledge problem. Retrieval can be more appropriate when a system must cite changing documents or access a private corpus. Fine-tuning can be useful for behavior, formatting, or task patterns, but it does not ensure factual reliability.

## Evaluation

Create a representative evaluation set before making many changes. Include:
- ordinary cases;
- edge cases and ambiguous requests;
- questions not answerable from provided context;
- safety and prompt-injection cases when relevant;
- expected outcomes or a documented scoring rubric.

Review failure examples, not just the aggregate score. Model-based judges can help scale evaluation but may have biases and should not automatically be treated as ground truth.

## Exercises

1. Compare greedy decoding with two sampling settings on the same prompts.
2. Count input and output tokens for several representative tasks.
3. Write a prompt that explicitly handles missing information.
4. Build a small evaluation set before iterating on a prompt.
5. Decide whether a sample task calls for retrieval, fine-tuning, ordinary application code, or a combination.
