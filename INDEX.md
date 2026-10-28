# Repository Index

Use this page to jump directly to a topic. The ROADMAP.md file gives the suggested learning order.

## Start here

- [Repository overview](README.md)
- [Learning roadmap](ROADMAP.md)
- [Core glossary](01-foundations/glossary.md)
- [Mathematics primer](02-python-for-ml/math_for_ml.md)
- [How to add material](CONTRIBUTING.md)
- [Reproducible experiment template](10-experiments/experiment_template.md)

## Python and data work

- [Python for ML guide](02-python-for-ml/README.md)
- [NumPy fundamentals](02-python-for-ml/numpy_basics.py)
- [pandas cleaning and grouping](02-python-for-ml/pandas_basics.py)
- [Data quality inspection](02-python-for-ml/data_quality_check.py)
- [Basic plotting](02-python-for-ml/matplotlib_basics.py)

## Classical machine learning

- [Classical ML introduction](03-machine-learning/README.md)
- [Linear regression from scratch](03-machine-learning/linear_regression_from_scratch.py)
- [Logistic regression from scratch](03-machine-learning/logistic_regression_from_scratch.py)
- [k-nearest neighbors from scratch](03-machine-learning/knn_from_scratch.py)
- [Feature scaling and leakage](03-machine-learning/feature_scaling_demo.py)
- [Polynomial overfitting experiment](03-machine-learning/overfitting_demo.py)
- [Regression metrics](03-machine-learning/metrics.py)

## Evaluation

- [Evaluation concepts and leakage](04-evaluation/README.md)
- [Classification metrics](04-evaluation/classification_metrics.py)
- [Threshold and precision/recall trade-off](04-evaluation/threshold_tradeoff.py)
- [Model validation checklist](04-evaluation/model_validation_checklist.md)
- [Regression metric tests](04-evaluation/test_metrics.py)
- [Classification metric tests](04-evaluation/test_classification_metrics.py)

Run the unit tests from the repository root:

    python -m unittest discover -s 04-evaluation -p 'test_*.py'

## Deep learning and transformers

- [Deep learning foundations](05-deep-learning/README.md)
- [Gradient descent in one dimension](05-deep-learning/gradient_descent_1d.py)
- [Tiny neural network with backpropagation](05-deep-learning/tiny_neural_network_numpy.py)
- [MNIST handwriting-recognition notebook](05-deep-learning/notebooks/MNIST_Neural_Network_From_Scratch.ipynb)
- [MNIST NumPy implementation](05-deep-learning/mnist_numpy_nn.py)
- [Deep MNIST implementation and backpropagation guide](05-deep-learning/mnist_neural_network_from_scratch.md)
- [Optional MNIST requirements](05-deep-learning/requirements-mnist.txt)
- [MNIST network tests and finite-difference gradient checks](05-deep-learning/test_mnist_numpy_nn.py)
- [Transformer and attention overview](06-transformers/README.md)
- [Self-attention with NumPy](06-transformers/self_attention_numpy.py)
- [Attention exercises](06-transformers/attention_exercises.md)
- [Transformer tensor shapes](06-transformers/parameter_shapes.md)
- [Tokenization and embeddings](06-transformers/tokenization_and_embeddings.md)

## Modern AI systems

- [LLM engineering](07-llm-engineering/README.md)
- [Prompt evaluation template](07-llm-engineering/prompt_evaluation_template.md)
- [LLM context-budget helper](07-llm-engineering/context_budget.py)
- [LLM application reliability checklist](07-llm-engineering/production_checklist.md)
- [RAG pipeline and failure modes](08-rag/README.md)
- [Text chunking example](08-rag/chunking_demo.py)
- [TF-IDF retrieval demo](08-rag/tfidf_retrieval_demo.py)
- [AI agents and tool safety](09-ai-agents/README.md)
- [Bounded allowlisted tool loop](09-ai-agents/safe_tool_loop.py)

As the repository grows, add each new lesson here and link it from the relevant section README.
