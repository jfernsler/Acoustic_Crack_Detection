import numpy as np
from dnn.base_classes import Layer

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
        self.eps = np.finfo(float).eps

    def __handleDataIn(self, dataIn):
        if len(dataIn.shape) == 1:
            dataIn = np.array([dataIn])
        return dataIn

    def forward(self, dataIn):
        dataIn = self.__handleDataIn(dataIn)
        self.setPrevIn(dataIn)
        Y =  np.tanh(dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        return 1 - np.tanh(self.getPrevOut()) ** 2

    def backward(self, gradIn, eta):
        return np.multiply(gradIn, self.gradient())