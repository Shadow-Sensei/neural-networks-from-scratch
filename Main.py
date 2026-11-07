import numpy as np 
from mnist_loader import load_mnist

'''Last comment was made here!'''
def ReLU(linear_input):
    return np.maximum(0, linear_input)

rng1 = np.random.default_rng(42)

Weights = rng1.standard_normal((16,784))

W1 = (np.sqrt(2 / 784) * Weights)

B1 = np.zeros(16)
#X = np.ones(784)



Weights2 = rng1.standard_normal((16,16))
W2 = (np.sqrt(2 / 16) * Weights2)
B2 = np.zeros(16)

W3 = rng1.standard_normal((10,16)) * np.sqrt(1 / 16)
B3 = np.zeros(10)


'''New vectorized implementation of the cross-entropy loss'''
def Cross_entropy(probabilities, correct_labels):
    batch_size = probabilities.shape[0]
    correct_probs = probabilities[np.arange(batch_size), correct_labels]
    correct_probs = np.clip(correct_probs, 1e-12, 1.0)
    Loss = -np.mean(np.log(correct_probs))
    return Loss


'''Reimplementation of the softmax function '''

def softmax_func(Z3):
            row_max = np.max(Z3, axis=1, keepdims=True)
            shifted_Z3 = Z3 - row_max
            exp_values = np.exp(shifted_Z3)
            sum_values = np.sum(exp_values, axis=1, keepdims=True)
            probabilities = exp_values / sum_values
            return probabilities


X_train,Y_train = load_mnist()


EPOCHS =5 
BATCH_SIZE = 64
LEARNING_RATE = 0.01

for epoch in range(EPOCHS):

    # Shuffle at the beginning of each epoch
    indices = np.random.permutation(len(X_train))

    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]

    epoch_loss = 0

    for start in range(0, len(X_shuffled), BATCH_SIZE):

        X_batch = X_shuffled[start:start + BATCH_SIZE]
        Y_batch = Y_shuffled[start:start + BATCH_SIZE]

        '''Vectorized Forward pass'''
        Z1 = X_batch @ W1.T + B1
        A1 = ReLU(Z1)

        Z2 = A1 @ W2.T + B2 
        A2 = ReLU(Z2)

        Z3 = A2 @ W3.T + B3

        P = softmax_func(Z3=Z3)


        Y_one_hot = np.zeros((len(Y_batch), 10))
        Y_one_hot[np.arange(len(Y_batch)), Y_batch] = 1


        batch_loss = Cross_entropy(P, Y_batch)
        epoch_loss += batch_loss * len(Y_batch)



        '''Backpropagation code '''
        '''Layer 3'''
        delta3 = (P - Y_one_hot) / X_batch.shape[0]  # '''dL/dZ3'''
        dW3 = delta3.T @ A2 #  '''dL/dW3'''
        dB3 = np.sum(delta3, axis=0) # '''dL/dB3'''

        '''Layer 2 '''
        dA2 = delta3 @ W3 #  '''dL/dA2'''
        delta2 = dA2 * (Z2 > 0) # '''dA2/dZ2'''
        dW2 = delta2.T @ A1 #  '''dL/dW2 '''
        dB2 = np.sum(delta2, axis = 0) # '''dL/dB2'''

        '''Layer 1 '''
        dA1 = delta2 @ W2 # dL/dA1
        delta1 = dA1 * (Z1 > 0 ) # dA1/dZ2
        dW1 = delta1.T @ X_batch 
        dB1 = np.sum(delta1, axis = 0)


        '''Holy Gradient Descent '''
        

        '''First Layer'''
        W1 -= LEARNING_RATE * dW1
        B1 -= LEARNING_RATE * dB1

        '''Second Layer'''
        W2 -= LEARNING_RATE * dW2
        B2 -= LEARNING_RATE * dB2

        '''Third Layer '''  
        W3 -= LEARNING_RATE * dW3   
        B3 -= LEARNING_RATE * dB3

    print(f"Epoch {epoch + 1}/{EPOCHS} completed!")
    epoch_loss /= len(X_train)

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f}")


np.savez(
    "mnist_model-5.npz",
    W1=W1,
    B1=B1,
    W2=W2,
    B2=B2,
    W3=W3,
    B3=B3
)

print("Model saved successfully!")