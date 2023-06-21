import imp
import numpy as np
from dnn.base_classes import Objective

class CrossEntropy(Objective):
    def __init__(self):
        super().__init__()
        
    def eval(self, y, yhat):
        return -1.0*np.sum(y*np.log(yhat + self.eps))/y.shape[0]

    def gradient(self, y, yhat):
        return -(y/(yhat + self.eps))