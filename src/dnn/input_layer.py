import numpy as np
from ..dnn.base_classes import Layer


# Input Layer ##
class InputLayer(Layer):
    """
    A class to represent an InputLayer.
    ...
    Attributes
    ----------
    X : numpy matrix
        Input values

    Methods
    -------
    setStdX(X)
        Sets the computed standard deviation of the input matrix

    setMeanX(X)
        Sets the computed mean of the input matrix

    getStdX()
        Returns the computed standard deviation of the input matrix

    getMeanX()
        Returns the computed mean of the input matrix

    forward(X)
        Returns the z-scored version of the input matrix
        (X-meanX)/stdX

    gradient()
        Unimplemented
    """

    def __init__(self, X):
        """
        Constructor
        Calculates and sets the mean and standard deviation

        Parameters
        ----------
            __stdX : float
                Standard deviation of the input array
            __meanX : str
                Mean of the input array
        """
        super().__init__()
        self.__stdX = []
        self.__meanX = []
        self.setPrevIn(X)
        self.setStdX(X)
        self.setMeanX(X)

    def setStdX(self, X):
        """Calculates and sets the standard deviation"""
        self.__stdX = np.std(X, axis=0)
        # wherever the standard deviation is 0, replace with 1
        self.__stdX[self.__stdX == 0]  = 1

    def setMeanX(self, X):
        """Calculates and sets the mean"""
        self.__meanX = np.mean(X, axis=0)

    def getStdX(self):
        """Returns the calcuated standard deviation"""
        return self.__stdX

    def getMeanX(self):
        """Returns the calculated mean"""
        return self.__meanX

    def forward(self, X):
        """
        Generates the z-scored values of the input array

        Parameters
        ----------
        X : Input array

        Returns
        -------
        z-scored array
        """
        self.setPrevIn(X)
        z_score = np.divide(np.subtract(X, self.__meanX), self.__stdX)
        self.setPrevOut(z_score)
        return z_score

    def gradient(self):
        """Unimplemented"""
        pass
