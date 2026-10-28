# 08 — Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation combines a retrieval step with a generative model. The application searches a collection for potentially relevant material and supplies selected passages to the model as context.

RAG can make answers more grounded in a specific collection, but it does not guarantee correct retrieval, complete evidence, or faithful citations.

## The pipeline

1. **Ingest:** collect authorized documents.
2. **Parse:** extract useful text and metadata while preserving source identity.
3. **Chunk:** split long documents into manageable segments.
4. **Embed:** map chunks to vectors using an embedding model.
5. **Index:** make vectors and metadata searchable.
6. **Retrieve:** find candidate passages for a user query.
7. **Rerank (optional):** reorder candidates using a more targeted relevance model.
8. **Assemble context:** select passages that fit the task and token budget.
9. **Generate:** instruct the LLM to answer using supplied evidence and acknowledge gaps.
10. **Evaluate:** check retrieval quality, answer quality, citation support, latency, and cost.

The best chunk size and overlap depend on document structure, model, retrieval method, and question type. They should be tested rather than chosen as universal constants.

## Sparse and dense retrieval

- **Sparse retrieval**, such as term-based ranking, can be effective when exact words and identifiers matter.
- **Dense retrieval** compares vector representations and can find semantically related passages even when wording differs.
- **Hybrid retrieval** combines signals, potentially improving coverage but adding complexity.

Metadata filters can constrain retrieval by document type, time, tenant, or permissions. Access checks must happen before exposing retrieved content to a model or user.

## Evaluation should separate stages

### Retrieval metrics

- Recall at k: did the relevant evidence appear in the first k retrieved results?
- Precision at k: how much of the retrieved set was relevant?
- Mean reciprocal rank: how highly was the first relevant result ranked?

### Answer metrics

- Correctness with respect to trusted evidence.
- Completeness without unsupported additions.
- Whether claims are supported by the cited passages.
- Whether the system appropriately admits missing evidence.

High retrieval recall does not guarantee a good answer; poor retrieval can make a good generator fail. Evaluate both stages.

## Common failure modes

- Relevant text was never ingested or parsed correctly.
- Chunk boundaries separated a fact from its context.
- Embedding similarity retrieved a passage that is topically related but does not answer the question.
- Too much context obscures the important evidence.
- The model answers from prior knowledge instead of the supplied sources.
- Citations point to documents that do not support the associated claim.
- Private documents leak because authorization was not applied during retrieval.
- A retrieved passage contains hostile instructions that attempt to manipulate the model.

Retrieved text should be treated as data, not trusted instructions. Delimit it clearly and keep tool permissions and authorization checks outside the model's control.

## Exercises

1. Build a small collection of 10 to 20 documents and write questions with known evidence.
2. Compare sparse and dense retrieval on exact-name and paraphrased questions.
3. Create an unanswerable question and verify that the application admits it cannot find evidence.
4. Measure recall at k before changing the generator prompt.
5. Inspect each citation and check whether it entails the claim it is attached to.
