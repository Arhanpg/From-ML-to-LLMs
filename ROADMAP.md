# Learning Roadmap: From ML to LLMs

Work through these stages in order. You can start coding early and return to mathematical details as the ideas become more complex.

## 00 — Foundations

- [ ] Distinguish AI, ML, deep learning, generative AI, and LLMs.
- [ ] Understand datasets, features, labels, parameters, and hyperparameters.
- [ ] Explain supervised, unsupervised, and reinforcement learning at a high level.
- [ ] Define a baseline and a useful success metric.
- [ ] Understand generalization, overfitting, underfitting, and data leakage.

Checkpoint: explain a typical ML project lifecycle without relying on library names.

## 01 — Python for data work

- [ ] Functions, modules, exceptions, and virtual environments.
- [ ] NumPy arrays, shapes, indexing, broadcasting, vectorization, and random generators.
- [ ] pandas tables, missing values, filtering, grouping, and joins.
- [ ] Plot distributions, relationships, and errors.
- [ ] Load a CSV and write a data-quality summary.

Checkpoint: take a small tabular dataset from raw input to a clean, documented summary.

## 02 — Mathematics for ML

- [ ] Vectors, matrices, dot products, norms, and matrix multiplication.
- [ ] Mean, variance, probability, and conditional probability.
- [ ] Common distributions and sampling.
- [ ] Derivatives, gradients, and the chain rule.
- [ ] Loss functions and gradient descent.

Checkpoint: explain how a gradient informs a parameter update.

## 03 — Classical machine learning

- [ ] Linear and logistic regression.
- [ ] k-nearest neighbors and distance metrics.
- [ ] Decision trees, random forests, and boosting.
- [ ] Clustering and dimensionality reduction.
- [ ] Feature engineering and preprocessing pipelines.

Checkpoint: train a baseline, compare it with another model, and explain the performance difference.

## 04 — Evaluation and experiments

- [ ] Training, validation, and test splits.
- [ ] Regression: MAE, MSE, RMSE, and R-squared.
- [ ] Classification: precision, recall, F1, confusion matrix, and ROC-AUC.
- [ ] Cross-validation, imbalance, and threshold selection.
- [ ] Leakage, distribution shift, and reproducibility.

Checkpoint: defend a metric choice and identify a way it could mislead you.

## 05 — Deep learning

- [ ] Perceptrons, layers, activations, and parameter counts.
- [ ] Forward propagation, loss, and backpropagation.
- [ ] SGD, momentum, Adam, learning rates, and regularization.
- [ ] MLPs, CNNs, sequence models, and embeddings.
- [ ] Training curves, checkpoints, and GPU-memory basics.

Checkpoint: explain one training step from input batch to parameter update.

### First hands-on deep-learning project: MNIST

After learning the basics of dense layers, activations, and gradient descent, study the repository's [MNIST NumPy network](05-deep-learning/mnist_numpy_nn.py), [annotated notebook](05-deep-learning/notebooks/MNIST_Neural_Network_From_Scratch.ipynb), and [deep backpropagation walkthrough](05-deep-learning/mnist_neural_network_from_scratch.md). The project connects tensor shapes, forward propagation, softmax with cross-entropy, the chain rule, ReLU's derivative, parameter gradients, and held-out test evaluation.

Before moving on, try to explain each gradient's shape and use the provided finite-difference tests to check selected derivatives.

## 06 — NLP and transformers

- [ ] Tokenization, token IDs, and embeddings.
- [ ] Query, key, and value vectors.
- [ ] Self-attention and multi-head attention.
- [ ] Residual connections, normalization, and feed-forward blocks.
- [ ] Encoder-only, decoder-only, and encoder-decoder models.

Checkpoint: trace text through tokenization, attention, and output logits.

## 07 — LLM engineering

- [ ] Next-token prediction and context windows.
- [ ] Prompt structure and structured outputs.
- [ ] Temperature, top-k, and top-p sampling.
- [ ] Pretraining, supervised fine-tuning, and preference optimization.
- [ ] LoRA/QLoRA, quantization, and inference trade-offs.
- [ ] Evaluation, hallucination analysis, and task-specific test sets.

Checkpoint: compare two inference configurations on the same test cases.

## 08 — Retrieval-augmented generation

- [ ] Document ingestion and chunking.
- [ ] Embeddings and semantic similarity.
- [ ] Vector indexes and metadata filters.
- [ ] Retrieval, reranking, and context assembly.
- [ ] Citation checks and retrieval evaluation.
- [ ] Privacy, access control, and prompt-injection risks.

Checkpoint: build a small RAG prototype and test questions the sources cannot answer.

## 09 — Agents and reliable systems

- [ ] Tools and typed inputs/outputs.
- [ ] State, planning, control flow, and stopping conditions.
- [ ] Single-agent workflows versus multi-agent systems.
- [ ] Repeatable tests, traces, retries, and timeouts.
- [ ] Cost and latency budgets, human approval, and prompt-injection boundaries.

Checkpoint: demonstrate a workflow that handles a tool failure safely.

## Project ladder

1. Explore and clean one small dataset.
2. Train a baseline model and report test metrics.
3. Compare two models under the same evaluation setup.
4. Train a small neural network and inspect learning curves.
5. Visualize or implement a small attention calculation.
6. Build a RAG prototype with source references.
7. Create a repeatable evaluation harness.

Mark a topic complete when you can explain it, run or implement an example, state its assumptions, and solve a small variation independently. Reading alone is not the same as being able to apply the idea.
