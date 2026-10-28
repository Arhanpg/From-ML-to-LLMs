# Mathematics for Machine Learning: A Practical Primer

You do not need to memorize every formula before writing models. You do need to recognize the main objects and understand how operations affect them.

## 1. Vectors

A vector is an ordered collection of numbers. A feature vector might represent one example:

    x = [area, age, bedrooms]

Its length is the number of features. In NumPy, a one-dimensional array with three elements commonly has shape (3,).

A vector can have meaning tied to its units. Adding square metres to a person's age is usually not meaningful simply because both values are numbers.

## 2. Matrices

A matrix is a two-dimensional grid of numbers. A tabular feature matrix commonly has shape (number of examples, number of features).

If X has shape (n, d) and w has shape (d,), then X @ w produces n predictions. The inner dimensions must agree. Checking dimensions is one of the easiest ways to diagnose a linear-algebra bug.

## 3. Dot product

The dot product multiplies corresponding elements and sums the results:

    [a, b] dot [c, d] = a*c + b*d

A linear model uses a dot product between the input features and learned weights, then adds a bias.

## 4. Mean and variance

The mean describes a center. Variance describes average squared spread around that mean; standard deviation is its square root.

Standardization transforms a feature using a mean and standard deviation:

    standardized = (value - training_mean) / training_standard_deviation

This often helps optimization when features have very different scales. In predictive workflows, calculate these statistics using training data only, then reuse them on validation and test data. Do not fit preprocessing on the entire dataset before splitting.

## 5. Derivatives and gradients

A derivative describes how a scalar function changes as an input changes. A gradient collects partial derivatives with respect to several parameters.

For a loss function, a gradient indicates local directions of increase. Gradient descent typically takes a step in the negative-gradient direction:

    new_parameter = old_parameter - learning_rate * gradient

The learning rate scales the update. It is not learned as an ordinary model parameter in basic gradient descent.

## 6. Probability intuition

A probability lies between zero and one and represents uncertainty under a model or process. A probability distribution assigns probabilities across possible outcomes.

Conditional probability asks about one event given information about another. This matters in classification, calibration, Bayesian reasoning, and interpreting model scores.

A model output of 0.9 should be interpreted as a probability only when the model and its calibration justify that interpretation. Some classification models output scores that are not well-calibrated probabilities.

## 7. Why the loss function matters

A training algorithm needs a numerical objective. Mean squared error penalizes large residuals strongly because errors are squared. Mean absolute error penalizes residual magnitude linearly. Classification commonly uses cross-entropy losses.

The chosen objective influences which mistakes the training procedure prioritizes. Metric selection should also reflect the real application, not only mathematical convenience.

## Practice

1. Write the shape of a matrix with 12 examples and 5 features.
2. What shape results from multiplying that matrix by a weight vector of length 5?
3. Standardize the values 10, 20, and 30 by hand using their population mean and standard deviation.
4. Explain why a too-large learning rate may increase a loss.
5. Give an example where a calibrated probability is more useful than a raw score.
