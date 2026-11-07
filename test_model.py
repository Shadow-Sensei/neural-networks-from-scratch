import numpy as np 
from mnist_loader import load_mnist_test

model = np.load("mnist_model-5.npz")

W1 = model["W1"]
B1 = model["B1"]

W2 = model["W2"]
B2 = model["B2"]

W3 = model["W3"]
B3 = model["B3"]

X_test, Y_test = load_mnist_test()

Z1 = X_test @ W1.T + B1
A1 = np.maximum(0, Z1)

Z2 = A1 @ W2.T + B2
A2 = np.maximum(0, Z2)

Z3 = A2 @ W3.T + B3

shifted = Z3 - np.max(Z3, axis=1, keepdims=True)
exp_values = np.exp(shifted)

P = exp_values / np.sum(exp_values, axis=1, keepdims=True)

predictions = np.argmax(P, axis=1)


correct_predictions = np.sum(predictions == Y_test)

accuracy = correct_predictions / len(Y_test) * 100

print(f"Correct predictions: {correct_predictions}/10000")
print(f"Test Accuracy: {accuracy:.2f}%")