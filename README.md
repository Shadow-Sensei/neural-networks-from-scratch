# Neural Networks from Scratch — The Math Behind the Magic

A fully connected handwritten-digit classifier built with **NumPy**, without PyTorch or TensorFlow. This repository accompanies the **Neural Networks from Scratch** workshop hosted by Vertex — The Tech Club.

The project shows how weighted sums, activation functions, matrix multiplication, cross-entropy, gradient descent, and backpropagation work together to classify handwritten digits from the **MNIST** dataset.

## Network architecture

```
MNIST image (28 × 28 pixels)
          ↓ flatten and normalize
784 input values
          ↓
16 neurons — ReLU
          ↓
16 neurons — ReLU
          ↓
10 output scores — Softmax
          ↓
Predicted digit (0–9)
```

The network has **13,002 trainable parameters**:

| Layer | Weights | Biases |
| --- | ---: | ---: |
| 784 → 16 | 12,544 | 16 |
| 16 → 16 | 256 | 16 |
| 16 → 10 | 160 | 10 |
| **Total** | **12,960** | **42** |

Training uses mini-batches of 64 images, categorical cross-entropy loss, and gradient descent with manually implemented backpropagation. The training script initializes weights with a seeded NumPy generator and updates every layer's weights and biases.

## Reported MNIST test results

These are the results recorded during development using the 10,000-image MNIST test set:

| Saved model | Training epochs | Test accuracy |
| --- | ---: | ---: |
| `mnist_model.npz` | 10 | 93.10% |
| `mnist_model-2.npz` | 20 | 94.06% |
| `mnist_model-3.npz` | 30 | 94.40% |
| `mnist_model-5.npz` | 50 | **95.21%** |

*The checkpoints are separate saved models. Accuracy on photographs of handwriting can be lower than accuracy on MNIST test images.*

## Contents

```text
NeuralNetwork/
├── data/                       # Four MNIST IDX gzip files
├── sample_images/              # Example photographed digits
├── Main.py                     # Train and save a network
├── mnist_loader.py             # Read and normalize MNIST training/test sets
├── test_model.py               # Evaluate a saved model on 10,000 test images
├── predict_handwritten.py      # Predict a digit from a local photo
├── mnist_model.npz             # 10-epoch checkpoint
├── mnist_model-2.npz           # 20-epoch checkpoint
├── mnist_model-3.npz           # 30-epoch checkpoint
├── mnist_model-5.npz           # 50-epoch checkpoint
├── requirements.txt
└── README.md
```

Local virtual environments and Python caches are excluded from Git.

## Getting started

You need **Python 3.10 or newer** and Git.

```bash
git clone https://github.com/YOUR_USERNAME/neural-networks-from-scratch.git
cd neural-networks-from-scratch
python -m venv .venv
```

Activate the environment on Windows (Git Bash):

```bash
source .venv/Scripts/activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

**Dataset:** The repository includes all four MNIST `.gz` files inside `data/`, so no separate download or change to paths is needed. Run commands from the repository's root directory.

## Evaluate the model

```bash
python test_model.py
```

The current `test_model.py` loads **`mnist_model-5.npz`** (the 50-epoch checkpoint). To evaluate another checkpoint, change the filename in this line of `test_model.py`:

```python
model = np.load("mnist_model-5.npz")
```

The script reports the number of correct classifications and test accuracy.

## Predict your own handwritten digit

Write **one dark digit on light paper**, take a clear photograph, and save it locally. Then run:

```bash
python predict_handwritten.py sample_images/digit.jpg --model mnist_model-5.npz --preview
```

Or, without opening a preview window:

```bash
python predict_handwritten.py sample_images/digit.jpg --model mnist_model-5.npz
```

The preprocessing step estimates the paper background, detects the ink, crops the digit, rescales and centers it on a 28×28 black canvas, and normalizes pixel values to the interval `[0, 1]`. The program prints a predicted digit and ten softmax probabilities.

**This is a file-based photo predictor**, not an automatic phone-to-laptop camera interface. Handwritten photos may be misclassified even when MNIST test accuracy is high. Softmax confidence is not a guaranteed probability that a prediction is correct.

## Train a new network

```bash
python Main.py
```

**Read this before training:** In the currently shared `Main.py`, `EPOCHS = 5` and the save filename is `mnist_model-5.npz`. Running the script unchanged will **overwrite the included 50-epoch checkpoint with a newly trained 5-epoch model**. Before running it, change the save filename (for example, to `my_new_model.npz`), and choose the number of epochs you want.

The training script uses a mini-batch size of 64 and a learning rate of 0.01. Its output includes mean loss per epoch. It saves the weights and biases (`W1`, `B1`, `W2`, `B2`, `W3`, `B3`) in an `.npz` file.

## The mathematics

- **Forward propagation:** each layer computes a weighted sum plus bias, then applies an activation function.
- **ReLU:** hidden-layer activation `max(0, z)`.
- **Softmax:** transforms the ten output scores into a distribution summing to 1.
- **Cross-entropy:** measures how much probability was assigned to the correct class.
- **Backpropagation:** applies the chain rule through the layers to compute gradients efficiently.
- **Gradient descent:** updates weights and biases in the direction that reduces the loss locally.

For the complete mathematical journey, refer to the accompanying workshop presentations and notes if they are shared separately.

## About this project

Created as a learning project and shared for students attending **Neural Networks from Scratch — The Math Behind the Magic**. The emphasis is understanding the mathematics and writing the core neural network using NumPy rather than relying on high-level machine-learning frameworks.

**Author:** Mohammed Tasowuff (Sufi) — Vertex, The Tech Club.

## License

No license has been added yet. Please contact the author about permissions beyond viewing the repository until a license is selected.
