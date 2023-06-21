import numpy as np
from scipy.signal import correlate2d, convolve2d
from dnn.base_classes import Layer


# Convolutional Layer
class ConvolutionalLayer(Layer):
    """
    A class to represent an ConvolutionalLayer.
    ...

    Methods
    -------
    forward(X)
    backward(grad_in, eta)
    """
    def __init__(self, in_channels, in_height, in_width, kernel_size, out_channels):
        super().__init__()
        
        self.in_shape = (in_channels, in_height, in_width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernels_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.out_shape = (out_channels, in_height - kernel_size + 1, in_width - kernel_size + 1)

        # initialize biases and kernels with random values
        self.biases = np.random.randn(*self.out_shape)
        self.kernels = np.random.randn(*self.kernels_shape)
        
    def forward(self, X):
        self.setPrevIn(X)
        output = np.copy(self.biases)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                output[i] += correlate2d(X[j], self.kernels[i, j], "valid")     
        self.setPrevOut(output)
        return output

    def gradient(self):
        ## shouldn't be called since backward is overriding base Layer
        print('danger!')
        return

    def __gradient_out(self, grad_in, kernel):
        return convolve2d(grad_in, kernel, "full")

    def __gradient_kernels(self, input, grad_in):
        return correlate2d(input, grad_in, "valid")

    def __update(self, grad_kernels, grad_in, eta):
        # update kernels and biases 
        self.kernels -= eta * grad_kernels
        self.biases -= eta * grad_in
    
    def backward(self, grad_in, eta):
        input = self.getPrevIn()
        
        # initialize gradient output and kernels
        grad_out = np.zeros(self.in_shape)
        grad_kernels = np.zeros(self.kernels_shape)

        # compute gredients for output and kernels
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                grad_out[j] += self.__gradient_out(grad_in[i], self.kernels[i, j])
                grad_kernels[i, j] = self.__gradient_kernels(input[j], grad_in[i])
    
        # update kernels and biases 
        self.__update(grad_kernels, grad_in, eta)

        return grad_out