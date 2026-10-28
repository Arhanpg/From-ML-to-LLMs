# MNIST Handwriting Recognition with a NumPy Neural Network

This lesson explains the first project in this learning path: classifying an image of a handwritten digit as one of the ten classes, 0 through 9, using a feedforward neural network whose forward pass and backpropagation are implemented with NumPy.

The implementation is intentionally small enough to inspect. The notebook uses TensorFlow/Keras only to download and load MNIST; the neural-network calculations and parameter updates are written directly with NumPy. It does **not** use Keras to build or train the model.

## 1. The problem being solved

MNIST contains grayscale images of handwritten digits. Each image is 28 pixels wide and 28 pixels high. The correct label is an integer from 0 to 9.

This is a **supervised, multi-class classification** task:

- **Supervised:** every training image comes with a known correct digit label.
- **Classification:** the answer is a category, not a continuous value such as temperature.
- **Multi-class:** there are ten mutually exclusive classes.
- **Training:** the network adjusts its weights and biases using labeled examples.
- **Inference:** after training, the network receives an image it has not trained on and returns a predicted digit.

The model does not start with a built-in understanding of handwriting. Its parameters begin with small random values. Training adjusts them so that output scores for the correct digits become larger relative to the alternatives.

## 2. Data loading and shapes

The notebook loads the dataset with:

~~~python
from tensorflow.keras.datasets import mnist

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
~~~

The notebook output shows these shapes:

- Training images: (60000, 28, 28)
- Test images: (10000, 28, 28)

Read these dimensions from left to right:

- 60,000 training images;
- each training image has 28 rows and 28 columns;
- 10,000 test images with the same image dimensions.

The labels are one integer per image. The training labels therefore contain 60,000 values, and the test labels contain 10,000.

The **training set** is used to adjust parameters. The **test set** should be held aside for the final evaluation on unseen examples. In the original notebook, the test images are used to show three example predictions, but the notebook does not calculate accuracy across the entire test set. The reported 90.06% is training accuracy, not test accuracy.

### What TensorFlow is doing here

The import from tensorflow.keras.datasets is a dataset-loading convenience. It provides the digit images and labels. The layers, activations, forward propagation, gradients, and gradient-descent update in the project are implemented by your own NumPy functions.

If you run this notebook in a fresh environment, TensorFlow must be installed for that dataset loader. See [requirements-mnist.txt](requirements-mnist.txt). The base repository requirements do not need to install TensorFlow for every learner.

## 3. Preprocessing: from images to feature vectors

A fully connected network expects a vector of features for each example. An image is initially a 28 by 28 grid, so the notebook flattens it into 784 numbers:

~~~python
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T / 255.0
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T / 255.0
~~~

### 3.1 Flattening

The dimensions 28 by 28 give 784 pixel positions because 28 × 28 = 784.

Before flattening, one image is a grid. After flattening, that image is represented by a one-dimensional list of 784 pixel intensities. The order is consistent, so a particular pixel position maps to the same feature index for every image.

The reshape operation starts with shape (number of images, 28, 28) and changes it to (number of images, 784). The transpose then changes the layout to (784, number of images).

Why put examples in columns? It makes the batch computations in this notebook convenient. With all images side by side in a matrix X, multiplying W1 by X computes the first-layer pre-activations for every image in one matrix multiplication.

### 3.2 Normalization

The raw grayscale pixel values are integers between 0 and 255. Dividing by 255.0 maps them into the interval from 0.0 to 1.0.

~~~text
normalized_pixel = raw_pixel / 255
~~~

This is simple scaling, not statistical standardization. It keeps the input magnitudes modest, which generally makes optimization easier than feeding raw values up to 255 into the same initial weights.

### 3.3 Important limitation of flattening

Flattening preserves the pixel values but removes the explicit two-dimensional grid structure. The network can learn associations between pixel positions, but it is not told directly that neighboring pixels are spatial neighbors. A convolutional neural network can encode that local spatial structure more naturally. This project uses a fully connected network because it is easier to implement and understand from scratch.

## 4. One-hot encoding: turning a digit into a target vector

The model has ten output neurons—one per digit. The label 7 is therefore represented as a vector of length ten whose position for class 7 is 1 and whose other positions are 0.

For example, the one-hot representation of class 3 is:

~~~text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
~~~

The notebook function is:

~~~python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T
~~~

The zero matrix is created first. Then, for each row i, the code writes a 1 into the column corresponding to that row's correct class. The transpose gives the final shape (10, number of examples), matching the model's convention that examples are columns.

The one-hot target is called Y in the equations below. The label for one example has a 1 for the correct class and 0 for every incorrect class.

## 5. The network architecture

The network has three computational stages:

~~~text
784 input features
        |
        v
Dense layer: 64 neurons + ReLU
        |
        v
Dense layer: 10 neurons + Softmax
        |
        v
10 class probabilities
~~~

It has one hidden layer. “Hidden” means it is between the input and output; it is not a statement that the layer's values are secret or inaccessible.

### 5.1 Parameters and their shapes

Let m be the number of images in the current batch. In the original notebook the whole training set is used in one batch, so m = 60,000.

| Symbol | Shape | Meaning |
|---|---:|---|
| X | (784, m) | Batch of flattened images |
| Y | (10, m) | One-hot class targets |
| W1 | (64, 784) | Input-to-hidden weights |
| b1 | (64, 1) | Hidden-layer biases |
| Z1 | (64, m) | Hidden pre-activations |
| A1 | (64, m) | ReLU hidden activations |
| W2 | (10, 64) | Hidden-to-output weights |
| b2 | (10, 1) | Output biases |
| Z2 | (10, m) | Output logits |
| A2 | (10, m) | Softmax class probabilities |

A weight matrix connects every neuron in one layer to every neuron in the next layer. W1 has 64 rows because there are 64 hidden neurons and 784 columns because every hidden neuron has a weight for each input pixel. W2 has 10 rows and 64 columns for the analogous hidden-to-output connections.

A bias is one adjustable offset per neuron. Its (neuron_count, 1) shape lets NumPy broadcast the same bias for that neuron across every example in the batch.

The model has 50,890 trainable scalar parameters: 64 × 784 weights and 64 hidden biases, plus 10 × 64 weights and 10 output biases.

## 6. Initialization: why the weights are not all zero

The notebook initializes weights from random normal distributions and biases as zeros:

~~~python
W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)
b1 = np.zeros((64, 1))
W2 = np.random.randn(10, 64) * np.sqrt(2.0 / 64)
b2 = np.zeros((10, 1))
~~~

If every neuron started with identical zero weights, neurons in the same layer would produce the same activations and receive identical gradients. They would fail to break symmetry and learn distinct roles. Randomized weights help break that symmetry.

The scale matters too. If activations or gradients grow or shrink dramatically through layers, learning can become unstable or ineffective. The factor sqrt(2 / fan_in) is commonly called **He initialization**, and is associated with ReLU networks. Here fan_in is the number of inputs feeding the layer.

The code applies this scale to both weight matrices. It is a reasonable educational initialization, but the output layer uses softmax rather than ReLU, and alternatives such as Xavier/Glorot initialization are also common there. The important lesson is that initialization is deliberate; it is not merely arbitrary noise.

The original code does not set a random seed before initialization. Exact training outputs can therefore differ between runs.

## 7. Forward propagation: computing a prediction

Forward propagation sends the batch from inputs through each layer to the predicted class probabilities. It does not update the parameters.

### 7.1 First dense layer

~~~text
Z1 = W1 X + b1
A1 = ReLU(Z1)
~~~

Matrix multiplication W1X combines pixel values with the learned weights. The result Z1 contains the pre-activation value for every hidden neuron and every image. Biases are added by broadcasting.

ReLU is the rectified linear unit:

~~~text
ReLU(z) = max(0, z)
~~~

A negative pre-activation becomes zero; a positive value passes through unchanged. ReLU introduces nonlinearity, which allows stacked layers to represent functions that a single linear transformation cannot represent.

### 7.2 Output dense layer

~~~text
Z2 = W2 A1 + b2
A2 = softmax(Z2)
~~~

Z2 contains ten **logits** per image. A logit is a raw, unnormalized score; it is not itself a probability.

Softmax converts the ten logits for each image into values that are non-negative and sum to 1:

~~~text
p_k = exp(z_k) / sum_j exp(z_j)
~~~

The implementation subtracts the maximum logit before exponentiation:

~~~python
exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
~~~

Subtracting the same maximum from all logits does not change softmax probabilities, because the common factor cancels from numerator and denominator. It does reduce the size of exponentials and helps avoid numerical overflow.

The subtraction and normalization use axis=0 because each column is one image and the ten rows are its classes. With keepdims=True, the result remains two-dimensional so broadcasting works as intended.

### 7.3 Why use argmax for the final class?

~~~python
np.argmax(A2, axis=0)
~~~

This finds the row with the highest probability for each image. Since each row corresponds to a class, that row index is the predicted digit. The output is a vector containing one predicted class per image.

Argmax chooses the largest score; it does not show whether the model is well-calibrated or certain in a broader statistical sense.

## 8. The objective: what should training improve?

The original notebook calculates gradients corresponding to **softmax plus categorical cross-entropy**, although it does not explicitly compute or print a cross-entropy loss.

For a single example with target vector y and output probabilities p, categorical cross-entropy is:

~~~text
L = - sum over classes k of y_k * log(p_k)
~~~

Because y is one-hot, all terms except the correct class are zero. If the correct class is c, then:

~~~text
L = -log(p_c)
~~~

This loss is small when the model assigns a high probability to the correct digit, and large when it assigns a very small probability to the correct digit. For m examples, the usual mean loss is the average of the individual losses.

Accuracy is different. Accuracy only checks whether argmax selected the correct class. A model can get a correct prediction with probabilities [0.51 for the correct class and 0.49 for another] or with [0.99 for the correct class and tiny probabilities elsewhere]; both count as one correct prediction. Cross-entropy distinguishes these cases and provides a smoother training objective.

## 9. Backpropagation — the central idea

**Forward propagation answers: “What prediction does this network make?” Backpropagation answers: “How should every weight and bias change to make the loss smaller?”**

Backpropagation is a systematic application of the **chain rule** from calculus. It begins with how the loss changes with the output logits, then propagates that error signal backward through the output layer, the hidden activation, and the input layer. It calculates gradients; it does not itself update the parameters. The update happens afterward in gradient descent.

### 9.1 The chain rule in plain language

Suppose a final answer depends on an intermediate quantity, and that intermediate quantity depends on a parameter. The parameter affects the final answer through the intermediate quantity. The chain rule multiplies the local rates of change along that path:

~~~text
change in loss / change in parameter
  = (change in loss / change in intermediate value)
    * (change in intermediate value / change in parameter)
~~~

A neural network is a composition of functions. The loss depends on the output probabilities; probabilities depend on logits; logits depend on weights and hidden activations; the hidden activations depend on earlier weights and inputs. The chain rule lets us combine these links efficiently.

A gradient is not a command to move in that direction. It says how the loss changes locally as a parameter increases. Gradient descent subtracts a scaled gradient because we seek to move downhill.

### 9.2 Step A — find the output-layer error signal

The output logits are Z2 and softmax produces A2. With categorical cross-entropy, the derivative with respect to each output logit has a particularly simple form:

~~~text
dZ2 = A2 - Y
~~~

Why does this work? For one example, softmax gives p_k = exp(z_k) / sum_j exp(z_j), while cross-entropy is L = -sum_k y_k log(p_k). Differentiating the combination of these expressions with respect to a logit z_j simplifies to p_j - y_j.

Therefore:

- if the predicted probability for a class is greater than its target value, its signal is positive;
- if the predicted probability is less than its target value, its signal is negative;
- for a one-hot target, the correct class is pushed upward while the incorrect classes are pushed downward.

For example, imagine the target is digit 3 and the predicted probabilities are [0.05, 0.05, 0.10, 0.60, 0.05, 0.03, 0.04, 0.03, 0.03, 0.02]. The target vector has 1 at class 3 and 0 elsewhere. Subtracting target from predictions gives a negative signal for class 3 (0.60 - 1 = -0.40) and positive signals for the incorrect classes. The downstream gradient-descent update uses these signals to make the correct logit more favorable and the incorrect logits less favorable, locally.

Notice a subtlety in the code: dZ2 is not divided by m immediately. The code divides by m when forming the weight and bias gradients. This is mathematically consistent because the layer's output error signals remain unaveraged until the parameter gradients are calculated.

### 9.3 Step B — compute gradients for W2 and b2

The output logits are:

~~~text
Z2 = W2 A1 + b2
~~~

For an individual connection, the weight's effect is proportional to the hidden activation feeding that connection. In matrix form, the batch gradient is:

~~~text
dW2 = (1/m) * dZ2 @ A1.T
db2 = (1/m) * sum(dZ2 across examples)
~~~

Shape checks make this less mysterious:

- dZ2 has shape (10, m).
- A1.T has shape (m, 64).
- Their product has shape (10, 64), exactly the shape of W2.
- Summing dZ2 across columns produces (10, 1), the shape of b2.

Why multiply by A1.T? For each output neuron, the gradient for each of its incoming weights accumulates the output error signal multiplied by the activation on that input connection. Matrix multiplication performs these products and sums over the batch efficiently.

Why average over m? The loss is a mean over examples. Dividing by m makes the gradient correspond to the average loss instead of a sum whose magnitude would grow with batch size.

For biases, each example contributes its output error signal directly, because the derivative of z = w*a + b with respect to b is 1. The batch contributions are summed and averaged.

### 9.4 Step C — carry the error backward into the hidden layer

The output layer depends on hidden activations through Z2 = W2 A1 + b2. The error signal with respect to hidden activations is:

~~~text
dA1 = W2.T @ dZ2
~~~

This is the chain rule in matrix form. Every hidden neuron can influence several output neurons. The output error signals are passed backward through the weights connecting hidden neurons to output neurons, and summed into one signal per hidden neuron and example.

The dimensions confirm the operation:

- W2.T has shape (64, 10).
- dZ2 has shape (10, m).
- dA1 has shape (64, m).

This says how the loss would change if the hidden activations changed. It is not yet the gradient with respect to the hidden pre-activations Z1, because ReLU stands between Z1 and A1.

### 9.5 Step D — pass through the ReLU derivative

The hidden activation is A1 = ReLU(Z1). ReLU is piecewise linear, so its derivative is:

~~~text
ReLU'(z) = 1 if z > 0
           0 if z < 0
~~~

At exactly zero, the derivative is not uniquely defined in classical calculus; implementations commonly choose 0 as a subgradient. This notebook uses zero there too, because its condition is Z1 > 0.

The hidden-layer error signal is therefore:

~~~text
dZ1 = dA1 elementwise-multiplied by indicator(Z1 > 0)
~~~

The multiplication is **elementwise**, not a matrix multiplication. NumPy's boolean array from ReLU_deriv becomes zeros and ones for this calculation.

Intuition: if a neuron's Z1 was negative, ReLU output was zero and the local derivative is zero. Its gradient through that activation is blocked for this step. If Z1 was positive, ReLU locally acts like the identity, so its derivative is one and the incoming gradient passes through.

This also explains why the original function takes Z1 rather than A1: the sign of the pre-activation tells us which side of the ReLU kink the unit was on.

### 9.6 Step E — compute gradients for W1 and b1

The first-layer pre-activations are:

~~~text
Z1 = W1 X + b1
~~~

The same dense-layer rule applies again:

~~~text
dW1 = (1/m) * dZ1 @ X.T
db1 = (1/m) * sum(dZ1 across examples)
~~~

Shapes:

- dZ1 is (64, m).
- X.T is (m, 784).
- dW1 becomes (64, 784), which matches W1.
- Summing dZ1 across examples gives (64, 1), matching b1.

For every input-to-hidden weight, the gradient combines the hidden-neuron error signal with the pixel value connected to that weight. A pixel that is zero for an image contributes zero to that image's weight-gradient product through this direct multiplication; a nonzero pixel contributes in proportion to its normalized intensity and the hidden neuron's error signal.

The gradient tells us how the mean loss changes if a parameter changes slightly. A positive gradient means increasing that parameter would locally increase the loss; gradient descent therefore subtracts a positive gradient. A negative gradient means subtracting the gradient increases the parameter, which locally decreases the loss.

### 9.7 Backpropagation at a glance

~~~text
Prediction:
X -> Z1 -> A1 -> Z2 -> A2

Backward gradient signals:
A2 - Y
   |
   v
dW2, db2
   |
   v  multiply by W2.T
dA1
   |
   v  multiply elementwise by ReLU'(Z1)
dZ1
   |
   v
dW1, db1

Parameter update:
W := W - learning_rate * dW
b := b - learning_rate * db
~~~

Every backward step answers a local question: how does the loss change with respect to this intermediate quantity? The chain rule connects the local answers to obtain gradients for earlier parameters.

## 10. Why the backward formulas look like matrix products

A dense layer takes Z = W A + b. For an individual output unit j and example i:

~~~text
z[j, i] = sum_k W[j, k] * A[k, i] + b[j]
~~~

The derivative of z[j, i] with respect to W[j, k] is A[k, i], the activation feeding that weight. By the chain rule, that example's contribution to the weight gradient is the output-side error signal multiplied by A[k, i].

Across all examples, summing these products is exactly the matrix product dZ @ A.T. Averaging gives the formulas used in the code. The bias gradient is the same reasoning without the activation term, because derivative of the bias contribution with respect to its bias is 1.

This same pattern appears in both dense layers. Only the relevant input activation and error-signal dimensions change.

## 11. Parameter updates: gradient descent

After backpropagation returns dW1, db1, dW2, and db2, the notebook updates each parameter:

~~~text
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
~~~

Here alpha is the **learning rate**. It determines the size of the step along the negative gradient.

- If the learning rate is extremely small, learning can be slow.
- If it is too large, an update can overshoot a good region or make training unstable.
- A moderate value can reduce loss effectively, but there is no universal best learning rate.

The update function is separate from backward propagation on purpose. Backprop computes gradients; gradient descent uses them to change parameters. Other optimizers, such as momentum-based methods and Adam, use gradients in different update rules.

## 12. The training loop, epoch, and batch size

The original training function runs 150 epochs:

~~~python
for i in range(1, epochs + 1):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    dW1, db1, dW2, db2 = backward_prop(
        Z1, A1, Z2, A2, W1, W2, X, Y
    )
    W1, b1, W2, b2 = update_params(
        W1, b1, W2, dW1, db1, dW2, db2, alpha
    )
~~~

The full 60,000-image training matrix is passed through the network at once. This is **full-batch gradient descent**: one update is calculated from all training examples per epoch. It is not mini-batch training.

An epoch means one pass over the complete training set. Because the notebook uses the full training set in every pass, each epoch uses all 60,000 images for one gradient update. Full-batch gradient descent is conceptually simple, but it can use significant memory and can be slower for large datasets. Mini-batch training splits data into smaller groups and performs multiple updates per epoch.

### A detail about the printed accuracy

The notebook computes A2, then updates parameters, then reports accuracy from the A2 calculated just before that update. Thus the displayed accuracy corresponds to the forward pass using the pre-update parameters for that iteration. The difference is only one update step, but in a polished training loop you would normally measure metrics after the update using a fresh forward pass, or clearly label the measurement as pre-update.

The notebook prints these training-accuracy values:

| Epoch | Training accuracy reported |
|---:|---:|
| 1 | 9.70% |
| 15 | 76.31% |
| 30 | 83.41% |
| 45 | 85.95% |
| 60 | 87.24% |
| 75 | 88.08% |
| 90 | 88.70% |
| 105 | 89.15% |
| 120 | 89.50% |
| 135 | 89.83% |
| 150 | 90.06% |

The rise from about 10% toward 90% indicates that the network is learning useful relationships between pixels and labels. Ten-class random guessing would be around 10% on a balanced task, but this does not make 90.06% a test score. It is the accuracy calculated on training examples.

## 13. Inference and plotting test examples

The notebook's inference function runs forward propagation and returns argmax classes:

~~~python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)
~~~

The plotting code selects a random test image, preserves its column shape with X_test[:, [sample_idx]], predicts a digit, then displays the original 28 by 28 image with predicted and real labels.

Using two-dimensional column indexing matters. X_test[:, sample_idx] would return a one-dimensional array of shape (784,), which does not match the matrix orientation expected by forward_prop. X_test[:, [sample_idx]] keeps shape (784, 1).

The test visualization is a useful sanity check, but three examples are not enough to estimate overall test performance. For that, evaluate predictions on all 10,000 test examples and compute:

~~~text
test_accuracy = number of correct test predictions / 10000
~~~

Do not use test accuracy as a repeated tuning signal for many architecture or hyperparameter changes. Use a validation split for those decisions and reserve the test set for final reporting.

## 14. What this first project teaches — and what it does not

### What it teaches

- Turning image tensors into a feature matrix.
- One-hot labels and multi-class targets.
- Matrix shapes and vectorized operations.
- Dense-layer forward propagation.
- ReLU and softmax.
- Deriving gradients using the chain rule.
- Backpropagating error signals through layers.
- Updating weights and biases with gradient descent.
- Distinguishing training from test examples.

### Limitations in the original notebook

1. It does not calculate or display cross-entropy loss, even though the gradient formula is the softmax/categorical-cross-entropy gradient.
2. It reports training accuracy, not full test accuracy.
3. It shows only three random test examples.
4. It uses one full training batch per update, so training can require considerable memory.
5. It does not set random seeds, so exact results are not guaranteed to reproduce.
6. It flattens images, ignoring the useful spatial bias of convolutional models.
7. It does not separately use a validation set to select the number of epochs or learning rate.
8. It uses the accuracy based on the forward pass computed just before the parameter update when reporting the epoch metric.

These points do not invalidate the project. They clarify its role: it is an educational implementation of a small network and its gradients, not a production-quality handwritten-digit service.

## 15. Suggested next experiments

Change one thing at a time and record the same evaluation metrics.

1. Add mean cross-entropy loss reporting and plot loss by epoch.
2. Evaluate the complete test set and report test accuracy separately from training accuracy.
3. Set a random seed so that initialization and example selection can be reproduced.
4. Compare learning rates such as 0.05, 0.15, and 0.3 using the same split and seed.
5. Compare 32, 64, and 128 hidden units.
6. Compare full-batch and mini-batch training.
7. Inspect a confusion matrix to see which digits are most often confused.
8. Compare the dense network with a small convolutional network, then explain why the architecture changes performance.
9. Implement a numerical gradient check on a tiny network: approximate a parameter's derivative with a small finite difference and compare it with backpropagation.
10. Save the parameters and verify that loading them reproduces predictions.

## 16. Backpropagation self-check

Try to answer these without looking at the previous sections:

1. Why is the derivative at the output logits A2 - Y for softmax with cross-entropy?
2. Why does dW2 use A1.T rather than X.T?
3. What does W2.T @ dZ2 represent?
4. Why is the ReLU derivative multiplied element by element?
5. Why do dW1 and dW2 have the same shapes as W1 and W2?
6. Why are the batch gradients averaged by m?
7. What is the difference between calculating a gradient and updating a parameter?
8. Why does the notebook's 90.06% result not establish 90.06% test accuracy?
9. Why is a random seed useful when comparing two learning-rate experiments?
10. What could go wrong if the learning rate is too large?

If you can answer these and reproduce the shapes of every intermediate matrix, you understand the core of this project rather than just being able to run the notebook.
