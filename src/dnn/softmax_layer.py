import numpy as np
from dnn.base_classes import Layer

class SoftmaxLayer(Layer):
    """
    A class for the Softmax activation layer.

    ...
    Methods
    -------
    forward(dataIn)
        Computes the Softmax function for the input data
          e^(z-max(z)) / sum_i(e^(z_i-max(z)))

    gradient()
        Calculate the partial derivative
    """

    def __init__(self):
        """Constructor"""
        super().__init__()

    def forward(self, dataIn):
        """
        Generate the Softmax values of the input array.

        Parameters
        ----------
        dataIn : Input array

        Returns
        -------
        Array with values processed by the Softmax function
        """
        self.setPrevIn(dataIn)
        exp = np.exp(dataIn - np.max(dataIn))
        Y = exp / exp.sum(axis=1, keepdims=True) 
        self.setPrevOut(Y)
        return Y

    def gradient(self):
        """ Calculate the gradient of softmax, and return in a tensor"""
        tensor=[]
        for h in self.getPrevOut():
            grad = np.identity(h.shape[0])
            for x,n in np.ndenumerate(grad):
                i = x[1]
                j = x[0]
                if i == j:
                    grad[i][j] = h[j] * (1 - h[j])
                else:
                    grad[i][j] = -h[i] * h[j]
            tensor.append(grad)
        grad_out = np.stack(tensor)
        return grad_out
