import numpy as np
from dnn.base_classes import Layer

# MS implementation
class ReLuLayer(Layer):
    def __init__(self):
        super().__init__()

    def __handleDataIn(self, dataIn):
        if len(dataIn.shape) == 1:
            dataIn = np.array([dataIn])
        return dataIn

    def forward(self, dataIn):
        dataIn = self.__handleDataIn(dataIn)
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        return np.where(self.getPrevIn() > 0, 1.0, 0.0)

    def backward(self, gradIn, eta):
        return np.multiply(gradIn, self.gradient())

epsilon = 0.0000001
# JF implementation
class ReLuLayer_jf(Layer):
    """
    A class for the ReLu activation layer.

    ...
    Methods
    -------
    forward(dataIn)
        Computes the ReLu function for the input data
        max(0,z)

    gradient()
        Unimplemented
    """

    def __init__(self):
        """Constructor"""
        super().__init__()

    def forward(self, dataIn):
        """
        Generate the ReLu values of the input array.

        Parameters
        ----------
        dataIn : Input array

        Returns
        -------
        Array with values processed by the ReLu function
        """
        self.setPrevIn(dataIn)
        Y = np.maximum(0, dataIn)
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        """ Calculate the gradient and return in as a row vector """
        tensor = []
        for h in self.getPrevIn():
            ident = np.identity(h.shape[0])
            for j,n in np.ndenumerate(h):
                if n < 0:
                    ident[j][j] = 0
                else:
                    ident[j][j] = 1
            tensor.append(ident)
        return np.stack(tensor)
