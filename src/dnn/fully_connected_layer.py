import numpy as np
from math import sqrt
from dnn.base_classes import Layer

# Fully Connected Layer ###
class FullyConnected(Layer):
    """
    A class to represent a fully connected layer.

    The in and out values are used to initialize weights and biases:
        weights = matrix of in=rows, out=columns
        bias = vector of out=length
    ...
    Attributes
    ----------
    in_count : int
        The number of input values for the fully connected layer

    out_count : int
        The number of output values for the fully connected Layer

    param_init : string
        Method of initilizing weights
        'default' : random values +-0.0001
        'xavier' : xavier et al method (based on input count)
        'he' : he et al method (based on input and output count)

    additional_features : int
        the number of additional features to be injected alongside
        each observation during forward propagation

    Methods
    -------
    setWeights(W)
        Set the weights value to the input W

    setBiases(b)
        Set the bias values to the input b

    getWeights()
        Returns the weights matrix

    getBiases()
        Returns the biases array

    set_adam_vals()
        Explicity set rho1 and rho2

    reset_adam_accumulators()
        set s and r back to zero

    forward(x)
        Returns the output of: xW + b

    gradient()
        Returns the transpose of W

    updateWeights_adam(gradIn, epoch, eta)
        Update layer weights using the ADAM method

    updateWeights(gradIn, eta)
        updates the layer's weights based on an incoming gradient
        and learning rate.
    """

    def __init__(self, in_count, out_count, param_init='default', additional_features=0, seed=0):
        """
        Constructor
        Calculate and set the mean and standard deviation.

        Parameters
        ----------
            __weights : numpy matrix
                Input weights for data
                initialized with random values +=10^4
            __biases : numpy array
                intialized with zeros - length of output
        """
        super().__init__()
        epsilon = 0.0001
        np.random.seed(seed)

        self.__additional_features = additional_features

        inputs = in_count + self.__additional_features
        self.__x2 = np.zeros(self.__additional_features)


        if param_init in ['xavier']:
            distro_range = sqrt(6./(inputs + out_count))
            self.__weights = epsilon * np.random.randn(inputs, out_count)
            self.__weights = np.random.uniform(size=(inputs, out_count),
                                               low = -distro_range,
                                               high = distro_range)
            self.__biases = np.random.uniform(size=(out_count),
                                               low = -distro_range,
                                               high = distro_range)
        elif param_init in ['he']:
            std_dev = sqrt(2./inputs)
            self.__weights = np.random.normal(0, std_dev, size=(inputs, out_count))
            self.__biases = np.random.normal(0, std_dev, size=(out_count))
        else:
            self.__weights = epsilon * np.random.randn(inputs, out_count)
            self.__weights = np.random.uniform(size=(inputs, out_count),
                                               low = -epsilon,
                                               high = epsilon)
            self.__biases = np.random.uniform(size=(out_count),
                                               low = -epsilon,
                                               high = epsilon)
        self.rho1 = 0.9
        self.rho2 = 0.99
        self.delta = 10**-8

        # weights
        self.s_w = 0
        self.r_w = 0
        # biases
        self.s_b = 0
        self.r_b = 0

    def setWeights(self, W):
        """Set weights with a numpy matrix."""
        self.__weights = W


    def setBiases(self, b):
        """Set biases with numpy array."""
        self.__biases = b


    def getWeights(self):
        """Return weights."""
        return self.__weights

    def getBiases(self):
        """Return biases."""
        return self.__biases

    def get_addtional_feature_count(self):
        """return number of extra features to be used"""
        return self.__additional_features

        
    def set_additional_features(self, x2):
        """Set injected features"""
        self.__x2 = x2


    def set_adam_vals(self, rho1=0.9, rho2=0.999, delta=10**-8):
        """Explicitly set rho1, rho2, delta."""
        self.rho1 = rho1
        self.rho2 = rho2
        self.delta = delta


    def reset_adam_accumulators(self):
        """Set s & r to zero."""
        # weights
        self.s_w = 0
        self.r_w = 0
        # biases
        self.s_b = 0
        self.r_b = 0


    def forward(self, x):
        """
        Generate the weighted and biased output of the layer.

        Parameters
        ----------
        x : Input array

        Returns
        -------
        weighted and biased array given by:
        xW + b
        """

        x_new = x

        if self.__additional_features > 0:
            try:
                x_new = np.concatenate((x, self.__x2), axis=1)
            except:
                print('Feature injection did not work in FCLayer')
                print(f'{x.shape} incompatible with {self.__x2.shape}')
                quit()

        self.setPrevIn(x_new)
        Y = np.add(np.dot(x_new, self.__weights), self.__biases)
        self.setPrevOut(Y)
        return Y


    def gradient(self):
        """Return the gradient - W^T"""
        return self.getWeights().T

    def backward(self, gradIn, eta):
        gradOut = super().backward(gradIn,eta)
        return gradOut

    def updateWeights_adam(self, gradIn, epoch, eta=0.001):
        pi = self.getPrevIn()
        po = self.getPrevOut()

        #get update for W
        dJdW = pi.T@gradIn
        #get update for b
        dJdb = np.sum(gradIn,0)

        ## first weights
        # update accumulators
        self.s_w = self.rho1*self.s_w+(1-self.rho1)*dJdW
        self.r_w = self.rho2*self.r_w+(1-self.rho2)*(dJdW*dJdW)

        # make our fractional parts
        adam_num_w = self.s_w / (1-self.rho1**epoch)
        adam_den_w = np.sqrt(self.r_w / (1-self.rho2**epoch)) + self.delta

        ## now bias
        # update accumulators
        self.s_b = self.rho1*self.s_b+(1-self.rho1)*dJdW
        self.r_b = self.rho2*self.r_b+(1-self.rho2)*(dJdW*dJdW)

        # make our fractional parts
        adam_num_b = self.s_b / (1-self.rho1**epoch)
        adam_den_b = np.sqrt(self.r_b / (1-self.rho2**epoch)) + self.delta

        # update weights
        self.__weights -= eta * adam_num_w / adam_den_w
        self.__biases -= eta * np.mean(adam_num_b / adam_den_b, axis=0)


    def updateWeights(self, gradIn, epoch=1, eta=0.001):
        pi = self.getPrevIn()
        po = self.getPrevOut()
        #get update for W
        dJdW = pi.T@gradIn
        #get update for b
        dJdb = np.sum(gradIn,0)

        #update weights
        self.__weights -= eta*dJdW/pi.shape[0]
        self.__biases -= eta*dJdb/pi.shape[0]
