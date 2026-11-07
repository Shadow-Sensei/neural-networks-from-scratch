# Neural Networks from Scratch
## Study notes and a guided tour of our NumPy MNIST classifier

**Companion reading for the “Neural Networks from Scratch — The Math Behind the Magic” workshop**  
**Audience:** Beginners who know a little Python and basic algebra  
**Project:** [neural-networks-from-scratch](https://github.com/Shadow-Sensei/neural-networks-from-scratch)

> **The central idea:** A neural network is a function with adjustable parameters. Training means changing those parameters so that its predictions become more accurate. Forward propagation computes an answer; a loss function measures its error; backpropagation computes how each parameter contributed to the error; gradient descent updates the parameters.

---

## 1. What problem are we solving?

We want a program that receives a picture of a handwritten digit and predicts one of ten classes: **0, 1, ..., 9**.

The **MNIST** dataset provides 60,000 training images and 10,000 test images. Each image is grayscale and has dimensions **28 × 28 pixels**. Each pixel is represented by a brightness value from 0 to 255.

We flatten the image into a vector with 784 entries, then divide by 255 so its values fall between 0 and 1. The image has not changed meaning: we have only changed its numerical representation.

```text
28 × 28 image → flatten → 784 numbers → neural network → 10 scores → prediction
```

**Questions to think about:** Why can't we directly use ordinary addition to recognize digits? What makes a slightly tilted 7 still recognizable as a 7?

## 2. The artificial neuron

A neuron takes inputs, multiplies each by a **weight**, adds a **bias**, and passes the result through an **activation function**.

```text
z = w1*x1 + w2*x2 + ... + wn*xn + b
activation = f(z)
```

- **Inputs (`x`)** carry information, such as pixel brightness or the activations of earlier neurons.
- **Weights (`w`)** determine how strongly different inputs influence a neuron. A negative weight can oppose a positive contribution.
- **Bias (`b`)** shifts the neuron's response; it lets the neuron respond even when its weighted sum is zero.
- **Activation (`f`)** determines the neuron's output and, when nonlinear, lets multilayer networks learn relationships that a single linear transformation cannot represent.

A neuron is not a little human brain. It is a mathematical operation, and much of the power comes from arranging many operations into layers and learning their parameters together.

## 3. Why do we need layers and nonlinear activation functions?

A layer is a collection of neurons computing outputs from the same incoming vector. A **dense (fully connected) layer** connects every entry of its input to every neuron in the layer.

Imagine applying two linear transformations in succession without any nonlinearity:

```text
h = W1*x + b1
out = W2*h + b2
    = (W2*W1)*x + (W2*b1 + b2)
```

They collapse into one linear transformation! Adding hidden layers without nonlinear activations would not provide the same expressive advantage.

Our hidden layers use **ReLU (Rectified Linear Unit)**:

```text
ReLU(z) = max(0, z)
```

ReLU returns zero for negative inputs and leaves positive inputs unchanged. The resulting network becomes a nonlinear function of its input.

At the output we use **softmax**, which converts ten scores (called **logits**) to nonnegative values that sum to one. The class with the highest score/probability is our prediction.

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

We subtract the largest logit before exponentiating to avoid unnecessarily large exponentials. This does **not** change the softmax result: multiplying all exponentials by the same factor cancels between numerator and denominator.

**Important:** Softmax values are model probabilities, not a guarantee that the prediction is correct or that the confidence is well calibrated.

## 4. Our exact network architecture

```text
Input                  Hidden layer 1       Hidden layer 2      Output
784 pixel values   →   16 ReLU neurons  →   16 ReLU neurons  →  10 logits
                                                                  ↓
                                                               softmax
                                                                  ↓
                                                              digit 0–9
```

Each weight matrix has one row per neuron in its destination layer:

| Parameter | Shape | Meaning |
| --- | --- | --- |
| `W1` | `(16, 784)` | Input → hidden layer 1 |
| `B1` | `(16,)` | Hidden layer 1 biases |
| `W2` | `(16, 16)` | Hidden layer 1 → hidden layer 2 |
| `B2` | `(16,)` | Hidden layer 2 biases |
| `W3` | `(10, 16)` | Hidden layer 2 → output |
| `B3` | `(10,)` | Output biases |

**Parameter count:**

```text
Layer 1: 16*784 + 16 = 12,560
Layer 2: 16*16  + 16 =    272
Layer 3: 10*16  + 10 =    170
TOTAL                      13,002 trainable parameters
```

A **parameter** is a learned weight or bias, not a hyperparameter such as the learning rate or the batch size.

## 5. A quick primer on shapes and matrix multiplication

For a batch of 64 images, `X_batch` has shape `(64, 784)`: **64 rows of images, 784 columns of pixel values**.

Our weight arrays are stored as `(number_of_neurons, number_of_inputs)`. Therefore, the code uses the transpose `.T` when multiplying a whole batch:

```python
Z1 = X_batch @ W1.T + B1       # (64,784) @ (784,16) -> (64,16)
A1 = np.maximum(0, Z1)        # (64,16)
Z2 = A1 @ W2.T + B2            # (64,16)  @ (16,16) -> (64,16)
A2 = np.maximum(0, Z2)        # (64,16)
Z3 = A2 @ W3.T + B3            # (64,16)  @ (16,10) -> (64,10)
```

The bias arrays are added to each row through **NumPy broadcasting**. The final array has ten scores per image.

If shapes seem mysterious, write down the matrix dimensions before running the code. For `(m, n) @ (n, p)`, the result is `(m, p)`.

## 6. Forward propagation: making predictions

Forward propagation passes an input through the network using its current weights and biases.

```text
Z1 = X * W1ᵀ + B1          A1 = ReLU(Z1)
Z2 = A1 * W2ᵀ + B2         A2 = ReLU(Z2)
Z3 = A2 * W3ᵀ + B3         P  = softmax(Z3)
```

`*` here means matrix multiplication when dimensions are matrices; the actual NumPy operator is `@`.

At the beginning of training, the weights are random, so predictions are generally poor. The purpose of training is to improve these predictions through repeated updates.

### Why initialize weights randomly?

If every neuron in a hidden layer starts with identical weights and biases, those neurons can receive identical gradients and keep learning the same features. Random initialization helps break that symmetry.

Our first two layers use **He-style initialization**:

```python
W1 = rng.standard_normal((16, 784)) * np.sqrt(2 / 784)
W2 = rng.standard_normal((16, 16))  * np.sqrt(2 / 16)
```

The `2 / fan_in` factor is motivated by maintaining a useful scale of activations in networks using ReLU. Our output layer uses `sqrt(1 / 16)` rather than the same scaling.

The random generator uses seed `42` for the initial weights. **However, the code shuffles batches with the separate global `np.random.permutation`, so complete training reproducibility is not guaranteed by that seed alone.**

## 7. What does the loss function measure?

Suppose the true label is `3`. We want the probability assigned to class 3 to be high.

Our training code uses **categorical cross-entropy**:

```text
loss for one example = -log(probability assigned to the true class)
loss for a batch     = average of those per-example losses
```

If the correct class receives probability `0.9`, the loss is small. If it receives `0.01`, the loss is much larger.

```python
correct_probs = P[np.arange(batch_size), correct_labels]
correct_probs = np.clip(correct_probs, 1e-12, 1.0)
loss = -np.mean(np.log(correct_probs))
```

Clipping the values above zero helps avoid taking `log(0)` due to floating-point rounding. Notice that the loss measures **the prediction against the known answer**; training does not require guessing the label.

## 8. Gradient descent: how the network learns

The **gradient** of the loss with respect to a parameter tells us how a small change in that parameter would affect the loss locally.

For a parameter `w`, the basic gradient descent update is:

```text
w_new = w_old - learning_rate * (dLoss/dw)
```

The negative sign means we take a small step in the direction expected to **reduce** the loss. The **learning rate** determines how large that step is.

- Too small: training can be slow.
- Too large: the updates may overshoot or destabilize training.
- A useful value depends on the problem and parameter scaling.

Our script uses **mini-batch gradient descent** with batch size `64` and learning rate `0.01`. Each mini-batch provides an estimate of the gradient over the training data. More precisely, using a separately shuffled batch at each update is commonly described as mini-batch stochastic gradient descent (SGD).

## 9. Backpropagation: the chain rule, applied systematically

One weight deep in the network doesn't directly produce the final loss. Its effect passes through several intermediate quantities. **Backpropagation applies the chain rule from the output layer backward** so we can compute the gradients efficiently.

For softmax combined with cross-entropy, the derivative with respect to the output logits has a particularly useful form:

```text
delta3 = (P - Y_one_hot) / batch_size
```

Here `Y_one_hot` is a matrix of correct labels, represented with a `1` in the correct class and `0` elsewhere.

Our code then computes gradients for each layer:

```python
# Output layer
Delta3 = (P - Y_one_hot) / X_batch.shape[0]
dW3 = Delta3.T @ A2
dB3 = np.sum(Delta3, axis=0)

# Second hidden layer
Delta2 = (Delta3 @ W3) * (Z2 > 0)
dW2 = Delta2.T @ A1
dB2 = np.sum(Delta2, axis=0)

# First hidden layer
Delta1 = (Delta2 @ W2) * (Z1 > 0)
dW1 = Delta1.T @ X_batch
dB1 = np.sum(Delta1, axis=0)
```

The mask `(Z > 0)` is the derivative used for ReLU (at exactly zero, our implementation chooses derivative zero).

Why does the transpose appear in `dW3 = Delta3.T @ A2`? Because the weight gradient must have the same shape as `W3`, namely `(10,16)`:

```text
Delta3.T: (10,64)    A2: (64,16)    product: (10,16)
```

Finally, each parameter is updated:

```python
W3 -= LEARNING_RATE * dW3
B3 -= LEARNING_RATE * dB3
# ...similarly for layers 2 and 1
```

**Backpropagation is not the same thing as gradient descent.** Backpropagation *calculates* the gradients; gradient descent *uses* them to update parameters.

## 10. One complete training cycle

1. Read a mini-batch of normalized images and correct labels.
2. Run forward propagation to compute probabilities.
3. Compare predictions to the labels using cross-entropy.
4. Run backpropagation to compute each weight/bias gradient.
5. Update the parameters using the learning rate.
6. Repeat for all batches; that completes **one epoch**.
7. Shuffle the data and train for another epoch.

An **epoch** is one complete pass through the training set. A **batch** is the subset used for a single parameter update. An **iteration/update step** is one such update.

Our script reports average training loss after each epoch. A decreasing training loss is useful, but it is **not by itself proof of good generalization**. Performance on separate data matters too.

## 11. Data loading and testing in this project

The `mnist_loader.py` script reads the compressed MNIST **IDX** files with Python's built-in `gzip` module. It skips the binary headers, interprets the remaining bytes using NumPy, reshapes images to `(60000, 784)` for training and `(10000, 784)` for testing, and scales pixel values to `[0,1]`.

| Dataset | Images | Purpose |
| --- | ---: | --- |
| Training set | 60,000 | Update learned parameters |
| Test set | 10,000 | Measure performance on separate MNIST images |

The file `test_model.py` loads `mnist_model-5.npz` by default, runs inference on all 10,000 test images, takes `argmax` over the ten output probabilities, and calculates the percentage of labels predicted correctly.

### Results recorded during project development

| Checkpoint | Epochs used for the saved version | Reported MNIST test accuracy |
| --- | ---: | ---: |
| `mnist_model.npz` | 10 | 93.10% |
| `mnist_model-2.npz` | 20 | 94.06% |
| `mnist_model-3.npz` | 30 | 94.40% |
| `mnist_model-5.npz` | 50 | **95.21%** |

These are **reported results from the project**, not accuracies independently reproduced for these notes. 95.21% corresponds to 9,521 correct predictions out of 10,000 test images.

### Important checkpoint warning

The current uploaded `Main.py` sets `EPOCHS = 5` **but still saves to `mnist_model-5.npz`**. Running it as written can overwrite the existing 50-epoch checkpoint. If you want to experiment, change the save filename first, or make a backup.

## 12. Recognizing a photograph versus recognizing MNIST

Real photographs are different from the neat training images: lighting, shadows, background texture, pen color, digit scale, and camera perspective all matter.

Our `predict_handwritten.py` script takes a local photograph and:

1. Converts it to grayscale, accounting for phone-image orientation.
2. Estimates the paper brightness from pixels at the image border.
3. Looks for darker handwriting and crops around the detected ink.
4. Resizes the digit while preserving its proportions.
5. Centers it on a 28 × 28 dark canvas with a bright digit.
6. Normalizes its pixels and runs the saved network.
7. Prints the predicted digit and the probabilities for all ten classes.

To try a sample from the repository root:

```bash
python predict_handwritten.py sample_images/digit.jpg --model mnist_model-5.npz --preview
```

The optional `--preview` flag uses Matplotlib to display the processed image. You can omit it if you only want a terminal prediction.

**Don't confuse MNIST test accuracy with real-world photo accuracy.** The 95.21% figure measures classification on the MNIST test set, not on arbitrary camera photos. A model can score highly on MNIST and still struggle with photographs.

## 13. How to run the project

The repository includes all four compressed MNIST data files, so participants do not need to download the dataset separately.

From the repository root:

```bash
# Clone the project
git clone https://github.com/Shadow-Sensei/neural-networks-from-scratch.git
cd neural-networks-from-scratch

# Create a local environment (recommended)
python -m venv .venv

# Activate it on Windows Git Bash
source .venv/Scripts/activate

# Or, on Linux/macOS:
# source .venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Evaluate the saved 50-epoch checkpoint
python test_model.py

# Test a photographed handwritten digit
python predict_handwritten.py sample_images/digit.jpg --model mnist_model-5.npz --preview
```

If your system uses `python3` rather than `python`, use that command instead. The core training and testing code needs NumPy; Pillow is used for photographed digits, and Matplotlib is used only for previews.

**Training:** `python Main.py` creates a new set of weights and trains them; it is not necessary if you only want to evaluate the existing model. Remember the checkpoint-overwrite warning above before training.

## 14. Common misunderstandings

| Misconception | More accurate idea |
| --- | --- |
| “The network memorizes an if/else rule for every digit.” | It learns parameter values that form a function mapping pixels to class scores. |
| “A weight is a probability.” | A weight is a learned numerical coefficient; the output softmax creates probabilities. |
| “Softmax makes the pixels normalized.” | Pixel normalization and output softmax are completely different operations. |
| “Loss and accuracy are the same.” | Loss measures how predictions score against labels; accuracy counts correct classes. |
| “Backpropagation changes weights.” | Backpropagation computes gradients; the optimizer update changes weights. |
| “A 95% test score guarantees 95% on phone photos.” | Performance may change when the input distribution changes. |
| “More epochs always means a better model.” | Extra epochs can help, but validation performance can plateau or worsen. |

## 15. Exercises (try before looking at the answers)

1. A neuron has inputs `[2, -1]`, weights `[0.5, -3]`, and bias `-1`. Compute its `z` value and its ReLU output.
2. Why is a nonlinear activation useful between two dense layers?
3. If `X_batch` has shape `(32,784)`, what shape does `X_batch @ W1.T` have?
4. Calculate the number of weights and biases in the final layer alone.
5. If the model assigns the true class probability `0.01`, is its cross-entropy loss larger or smaller than when it assigns `0.8`?
6. What job does backpropagation do that forward propagation does not?
7. Why do we subtract the largest logit before applying `exp` in softmax?
8. What would you change before running `Main.py` so you don't overwrite the best checkpoint?
9. A model gets 9,521 out of 10,000 MNIST test images correct. What is its accuracy?
10. Why might a photo of your own handwriting fail even when MNIST test accuracy is high?

### Answers

1. `z = 2(0.5) + (-1)(-3) - 1 = 3`; `ReLU(3) = 3`.
2. Without nonlinearities, consecutive affine layers collapse into one affine transformation.
3. `(32,16)`.
4. `10 × 16 = 160` weights and `10` biases, for `170` parameters.
5. Larger: `-log(0.01)` is larger than `-log(0.8)`.
6. It calculates derivatives of the loss with respect to weights and biases using the chain rule.
7. Numerical stability; it leaves the resulting probability ratios unchanged.
8. Change the file passed to `np.savez`, or back up `mnist_model-5.npz`.
9. `(9521 / 10000) × 100 = 95.21%`.
10. Photos can differ from MNIST in lighting, scale, background, strokes, and perspective.

## 16. Where to go next

A good next learning path is to **understand every shape in the forward pass**, derive the gradient of softmax plus cross-entropy, implement a finite-difference gradient check, then compare the results of different hyperparameters (learning rate, batch size, hidden width) while keeping a separate validation set.

For deeper study:

- **3Blue1Brown — Neural Networks:** visual intuition for neurons, layers, gradient descent, and backpropagation.
- **Michael Nielsen — Neural Networks and Deep Learning:** a freely available book that develops the mathematics and a handwritten-digit classifier.
- **NumPy documentation:** broadcasting, `@` matrix multiplication, `argmax`, array shapes, and random-number generators.

### Final takeaway

A neural network is not magic and it doesn't learn by being told a huge list of handwritten-digit rules. It learns through a repeated mathematical process:

```text
INPUT → FORWARD PASS → LOSS → BACKPROPAGATION → PARAMETER UPDATE → REPEAT
```

Once you understand how each arrow works, the network stops being a black box and becomes a system you can inspect, debug, and improve.

---

*Prepared as additional workshop reading. Implementation-specific statements refer to the project scripts supplied for this workshop; accuracy figures are the results reported during project development.*
