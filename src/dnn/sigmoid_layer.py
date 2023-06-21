import numpy as np
from dnn.base_classes import Layer

class SigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        Y = self.sigmoid(dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        return self.getPrevOut() * (1.0 - self.getPrevOut())

    def backward(self, gradIn, eta):
        return np.multiply(gradIn, self.gradient())


epsilon = 0.0000001

class SigmoidLayer_jf(Layer):
    """
    A class for the Sigmoid activation layer.

    ...
    Methods
    -------
    forward(dataIn)
        Computes the Sigmoid function for the input data
        1/(1+e-z)

    gradient()
        Unimplemented
    """

    def __init__(self):
        """Constructor"""
        super().__init__()

    def forward(self, dataIn):
        """
        Generate the Sigmoid values of the input array.

        Parameters
        ----------
        dataIn : Input array

        Returns
        -------
        Array with values processed by the Sigmoid function
        """
        self.setPrevIn(dataIn)
        Y = np.divide(1,1 + np.exp(-dataIn))
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        """ Calculate the gradient and return in as a row vector """
        tensor=[]
        for h in self.getPrevOut():
            ident = np.identity(h.shape[0])
            tensor.append((np.multiply(h, np.subtract(1., h)) + epsilon)*ident)
        return np.stack(tensor)
