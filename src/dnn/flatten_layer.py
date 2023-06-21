import numpy as np
from dnn.base_classes import Layer

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        self.setPrevIn(X)
        self.grad_shape = X.shape
        total_size = X.shape[0]
        for i in range(1, len(X.shape)):
            total_size *= X.shape[i]

        self.flat_shape = (1, total_size)
        flatten = np.reshape(X, self.flat_shape)
        self.setPrevOut(flatten)
        return flatten

    def gradient(self):
        # shouldn't be called since backward overrides the base Layer
        return
    
    def backward(self, grad_in, eta):
        return np.reshape(grad_in, self.grad_shape)
        