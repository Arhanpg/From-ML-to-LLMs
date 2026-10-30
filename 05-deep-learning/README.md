# 05 — Deep Learning Foundations

Deep learning uses neural networks built from layers of parameterized operations. Training adjusts those parameters so predictions better match a target or objective.

## A simple neuron

A basic neuron computes a weighted sum and then applies an activation:

    z = w1*x1 + w2*x2 + ... + b
    output = activation(z)

The weights control how inputs contribute, and the bias shifts the value. An activation such as ReLU or sigmoid introduces nonlinearity. Stacking only linear operations without nonlinear activations is still equivalent to a single linear transformation, so depth alone would not provide the expressive behavior usually associated with deep networks.

## Main training loop

1. Obtain a batch of inputs and targets.
2. Run the forward pass to compute predictions.
3. Compute a loss from predictions and targets.
4. Compute gradients of the loss with respect to parameters.
5. Use an optimizer to update parameters.
6. Repeat over batches and epochs.
7. Evaluate on data not used to fit the parameters.

Backpropagation efficiently applies the chain rule to compute gradients through a composition of operations. Modern frameworks use automatic differentiation to calculate these gradients; the conceptual job is still to determine how changing each parameter changes the loss.

## Vocabulary

- **Epoch:** one pass through the training dataset.
- **Batch:** the examples used in one training update.
- **Learning rate:** controls update size.
- **Activation:** a function applied to a layer's pre-activation values.
- **Gradient:** direction and rate of local change of an objective with respect to parameters.
- **Optimizer:** the algorithm that uses gradients to update parameters.
- **Regularization:** methods intended to reduce undesirable model complexity or improve generalization.
- **Checkpoint:** a saved state, often including model parameters and sometimes optimizer state.

## Common architecture families

- **MLP:** fully connected layers; a useful baseline for fixed-size feature vectors.
- **CNN:** uses convolutional structure and shared filters, often useful for spatial patterns.
- **RNN and gated variants:** process sequences recurrently, with limitations that motivated other sequence architectures.
- **Transformer:** uses attention and feed-forward blocks to process token representations; central to many current language models.

Architecture choice should follow the data and task rather than fashion.

## Training diagnostics

Inspect training and validation loss across epochs.

- Both losses high: the model may be underfitting, or optimization/data may be inadequate.
- Training loss falls while validation loss worsens: overfitting is a possibility.
- Both losses are unstable: investigate learning rate, gradients, scale, data quality, and numerical issues.
- Very good results with suspiciously simple training: verify the split and investigate leakage before celebrating.

These are diagnostic hints, not proofs. Data size, noise, model capacity, and task difficulty all affect the curves.

## Recommended hands-on sequence

1. Compute a neuron output by hand.
2. Implement a tiny forward pass with NumPy.
3. Compare a model's analytic gradient with a finite-difference approximation.
4. Train a small model using a deep-learning framework.
5. Change one training hyperparameter and compare the learning curves.
6. Save a checkpoint and verify that loading it reproduces predictions.

## Check your understanding

1. Why do hidden layers usually need nonlinear activations?
2. What is the difference between an epoch and an optimization step?
3. Why can a large learning rate destabilize training?
4. What information does a validation curve provide?
5. Why is backpropagation not the same as the optimizer?


## First complete project: MNIST handwriting recognition

This project classifies handwritten digits (0–9) from 28 × 28 grayscale images using a dense neural network built with NumPy.

- [Annotated Jupyter notebook](notebooks/MNIST_Neural_Network_From_Scratch.ipynb) — the from-scratch model, detailed backpropagation derivation, vanilla gradient descent, NumPy Adam implementation, and code to evaluate both models on all 10,000 test images.
- [Deep implementation and backpropagation guide](mnist_neural_network_from_scratch.md) — preprocessing, parameter shapes, forward pass, softmax/cross-entropy derivative, ReLU derivative, full-batch gradient descent, calculus and stationary points, loss-function selection, Adam mathematics, and careful interpretation of the saved experiment.
- [Runnable NumPy implementation](mnist_numpy_nn.py) — reproducible initialization, mini-batch training, loss reporting, and separate full test-set metrics.
- [Optional MNIST requirements](requirements-mnist.txt).

Install the MNIST-specific dependencies from the repository root:

    python -m pip install -r 05-deep-learning/requirements-mnist.txt

Run the companion script:

    python 05-deep-learning/mnist_numpy_nn.py --epochs 5 --batch-size 256

The updated notebook compares **vanilla full-batch gradient descent** with a NumPy implementation of **Adam**. Its saved outputs show 89.88% and 96.14% respectively at epoch 150, but these are displayed **training accuracies**, not test accuracies. The notebook explains that the two runs use different learning rates and independent random initializations, and that the printed metric is calculated from the forward pass before that epoch's update. A final code cell recomputes train/test cross-entropy and accuracy for both trained parameter sets; run that cell to obtain the full test results.

The companion script uses shuffled mini-batches by default for a more practical local run, prints explicit cross-entropy loss, and evaluates the held-out test set after training. Its metrics need not match the full-batch notebook experiment.

### Key lesson

Backpropagation applies the chain rule to calculate gradients of the loss with respect to every weight and bias. Gradient descent uses those gradients to update the parameters. These are related but distinct operations.


Run the MNIST network's unit tests (these tests do not download MNIST or require TensorFlow):

    python -m unittest discover -s 05-deep-learning -p "test_mnist*.py"

The suite checks tensor shapes, softmax probability normalization, one-hot encoding, and selected analytical gradients against central finite differences. A passing gradient check provides evidence that the backward formulas match the loss for the tested tiny network; it does not prove every possible input or training configuration is correct.


## Why the notebook does not solve dy/dx = 0 directly

The derivative condition f'(x) = 0 identifies stationary points for suitable one-variable problems; it does not prove a point is a global minimum. A neural network has 50,890 parameters in this example, so a stationary point would require every partial derivative to be zero simultaneously. Those equations are coupled and the loss surface is generally non-convex. Backpropagation is still calculus: it uses the chain rule to calculate the gradients. The optimizer uses those gradients to make iterative parameter updates. The detailed guide explains why a zero gradient does not certify a global minimum or good test performance.

## Why cross-entropy rather than MAE/RMSE?

This model performs ten-class classification and outputs probabilities with softmax. Its output gradient, A2 - Y, is the standard derivative associated with softmax plus categorical cross-entropy and one-hot targets. The notebook does not print a cross-entropy loss during training, but that is the objective implied by the gradient formula. MAE and RMSE are common regression metrics; they can be applied to one-hot classification targets, but cross-entropy is generally the more suitable probability-based objective because it strongly penalizes assigning very low probability to the correct class. Accuracy is a separate metric, not the differentiable loss.

## What Adam changes

Vanilla gradient descent applies the current gradient directly. Adam maintains a moving average of gradients (first moment) and a moving average of squared gradients (second raw moment), corrects their initialization bias, and uses both to scale each parameter's update. The notebook implementation is **Adam, not AdamW**: it does not add decoupled weight decay. The saved Adam result is best treated as an observed training result from this run, not as a claim of 96.14% test accuracy or a guarantee that Adam always outperforms vanilla gradient descent.
