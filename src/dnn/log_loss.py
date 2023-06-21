import numpy as np
from dnn.base_classes import Objective

epsilon = 0.0000001

class LogLoss(Objective):
    """
    A class for the Log Loss objective layer.

    ...
    Methods
    -------
    eval(y, yhat)
        Computes returns the loss
        -(yln(yhat)+(1-y)ln(1-yhat))
        Add epison to log functions to avoid -inf

    gradient(y, yhat)
        Computes and returns the gradient
    """
    def __init__(self):
        super().__init__()

    def eval(self ,y, yhat):
        return -(y * np.log(yhat+epsilon) + (1-y) * np.log(1-yhat+epsilon)) / y.shape[0]

    def gradient(self ,y, yhat):
        return -(y-yhat) / (yhat * (1-yhat) + epsilon)
