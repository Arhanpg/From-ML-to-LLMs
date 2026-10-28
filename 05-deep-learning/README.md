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
