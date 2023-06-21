from abc import ABC, abstractmethod
import numpy as np

# Base Layer Class
class Layer(ABC):
    """
    Abstract base class for all deep layers
    """
    def __init__(self):
        self.eps = np.finfo(float).eps
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut

    # def backward(self, gradIn):
    #     return np.multiply(gradIn, self.gradient())

    def backward(self, gradIn, eta):
        sg = self.gradient()

        if len(sg.shape) == 2:
            return np.dot(gradIn, sg)

        grad = np.zeros((gradIn.shape[0],sg.shape[2]))

        for n in range(gradIn.shape[0]): #compute for each observation in batch
            # print(f'in backward gradIn: {gradIn.shape}')
            # print(f'in backward sg: {sg.shape}')
            grad[n,:] = np.dot(gradIn[n,:],sg[n,:,:])
        return grad

    @abstractmethod
    def forward(self, dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass


# Base Objective Class
class Objective(ABC):
    def __init__(self):
        self.eps = np.finfo(float).eps

    @abstractmethod
    def eval(self, y, yhat):
        pass

    @abstractmethod
    def gradient(self, y, yhat):
        pass
