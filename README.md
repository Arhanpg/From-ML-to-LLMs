# From ML to LLMs

A concept-first, hands-on learning repository for Artificial Intelligence, Machine Learning, Deep Learning, Large Language Models, and modern AI engineering.

The goal is not just to collect snippets. Each topic should help you understand what a concept does, why it works, when to use it, how to implement it, and how to test your understanding.

**Browse the lessons:** [INDEX.md](INDEX.md) · [ROADMAP.md](ROADMAP.md) · [CONTRIBUTING.md](CONTRIBUTING.md)

## What you will find

- Concept notes with intuition, terminology, assumptions, and limitations.
- Small runnable examples.
- Selected from-scratch implementations using NumPy.
- Experiments, comparisons, plots, and error analysis.
- Notes on transformers, LLMs, retrieval-augmented generation (RAG), and AI agents as the repository grows.

## Learning roadmap

Follow ROADMAP.md for the suggested order. The first end-to-end neural-network project is documented in [the MNIST backpropagation guide](05-deep-learning/mnist_neural_network_from_scratch.md), with an [annotated notebook](05-deep-learning/notebooks/MNIST_Neural_Network_From_Scratch.ipynb) and [runnable NumPy script](05-deep-learning/mnist_numpy_nn.py). The notebook now compares vanilla full-batch gradient descent with a NumPy Adam implementation, explains why the derivative condition is not a shortcut to a global minimum, and discusses why softmax classification uses cross-entropy rather than MAE/RMSE as its usual objective.

| Stage | Topics | Goal |
|---|---|---|
| 00 | AI and ML foundations | Build a reliable mental model |
| 01 | Python for data work | Work confidently with arrays and tables |
| 02 | Mathematics for ML | Connect math to models |
| 03 | Classical ML | Understand training and prediction |
| 04 | Evaluation | Measure performance and avoid misleading results |
| 05 | Deep learning | Understand neural networks and gradient-based training |
| 06 | NLP and transformers | Learn tokens, attention, and transformer blocks |
| 07 | LLM engineering | Explore prompting, fine-tuning, and inference |
| 08 | RAG | Build systems grounded in retrieved sources |
| 09 | Agents | Study tools, workflows, reliability, and safety |

## Getting started

Recommended: Python 3.10 or newer and a virtual environment.

Create a virtual environment:

    python -m venv .venv

Activate it:

- Windows PowerShell: .venv\Scripts\Activate.ps1
- macOS/Linux: source .venv/bin/activate

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

The MNIST notebook/script has an optional TensorFlow dependency for loading the dataset. Install it only when working on that lesson:

    python -m pip install -r 05-deep-learning/requirements-mnist.txt

Run examples from the repository root:

    python 02-python-for-ml/numpy_basics.py
    python 02-python-for-ml/pandas_basics.py
    python 03-machine-learning/linear_regression_from_scratch.py
    python 03-machine-learning/logistic_regression_from_scratch.py
    python 05-deep-learning/mnist_numpy_nn.py --epochs 5 --batch-size 256
    python 06-transformers/self_attention_numpy.py
    python 08-rag/tfidf_retrieval_demo.py

## How to study a topic

1. Read the explanation and restate it in your own words.
2. Run the example and predict the output first.
3. Change one input or parameter and observe the difference.
4. Explain the result and identify its assumptions.
5. Try a small exercise without looking at the solution.
6. Revisit the idea later and implement it from memory.

See CONTRIBUTING.md for the structure used for new lessons. Several examples use deterministic synthetic data, so you can run them without downloading datasets or configuring API keys.

Run the evaluation metric tests from the repository root:

    python -m unittest discover -s 04-evaluation -p 'test_*.py'

Run the neural-network tests, including finite-difference checks of backpropagation:

    python -m unittest discover -s 05-deep-learning -p 'test_mnist*.py'

## Repository principles

- Prefer clarity and correctness over cleverness.
- Explain inputs, outputs, shapes, and assumptions.
- Keep random examples reproducible with fixed seeds.
- Split data before fitting preprocessing or models.
- Never commit credentials, private datasets, or large generated artifacts.
- Distinguish a learning demonstration from production-ready code.

## License

The repository includes the license in LICENSE. Check its terms before reusing or distributing material.
