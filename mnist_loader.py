import gzip
import numpy as np 

def load_mnist():

    with gzip.open("data/train-images-idx3-ubyte.gz", "rb" ) as file:
        magic_bytes = file.read(16)
        image = file.read()
        array = np.frombuffer(image,dtype=np.uint8)
    X_train = array.reshape(60000,784)
    X_train = X_train/255


    with gzip.open("data/train-labels-idx1-ubyte.gz", "rb" ) as file:
        magic_bytes_labels = file.read(8)
        labels = file.read()
    Y_train = np.frombuffer(labels,dtype=np.uint8)

    return X_train,Y_train


X_train,Y_train = load_mnist()


def load_mnist_test():
    with gzip.open("data/t10k-images-idx3-ubyte.gz", "rb") as file:
        file.read(16)
        images = file.read()

    X_test = np.frombuffer(images, dtype=np.uint8)
    X_test = X_test.reshape(10000, 784)
    X_test = X_test.astype(np.float32) / 255.0

    with gzip.open("data/t10k-labels-idx1-ubyte.gz", "rb") as file:
        file.read(8)
        labels = file.read()

    Y_test = np.frombuffer(labels, dtype=np.uint8)

    return X_test, Y_test

#print(X_train.shape)
#print(Y_train.shape)
#for i in range(10):
 #   print(Y_train[i])

#print(np.max(X_train))
#print(np.min(X_train))

if __name__ == "__main__":
    X_train, Y_train = load_mnist()
    print(X_train.shape)
    print(Y_train.shape)# Code to run only when this file is executed directly