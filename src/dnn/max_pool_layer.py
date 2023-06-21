import numpy as np
from dnn.base_classes import Layer

# simple, but slow, non optimized version of 
# Max Pool layer with a pool size of 2.
class MaxPoolLayer(Layer):
    def __init__(self):
        super().__init__()
        pass
  
    def iterate(self, input):
        ch, h, w = input.shape
        new_h = h // 2
        new_w = w // 2
        for n in range(0, ch):
            for i in range(0, new_h):
                for j in range(0, new_w):
                    img = input[n, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    yield img, n, i, j

    def gradient(self):
        return super().gradient()

    def forward(self, input):
        '''
        forward pass of the max pool layer
        '''
        self.setPrevIn(input)
        ch, h, w = input.shape
        output = np.zeros((ch, h // 2, w // 2))
        for img, n, i, j in self.iterate(input):
            maxout = np.amax(img, axis=(0, 1))
            output[n, i, j] = maxout
        self.setPrevOut(output)
        return output

    def backward(self, grad_in, eta):
        '''
        backward pass of the max pool layer
        '''
        grad_out = np.zeros(self.getPrevIn().shape)
        for img, n, i, j in self.iterate(self.getPrevIn()):
            h, w = img.shape
            amax = np.amax(img, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    if img[i2, j2] == amax: # if an element of img was max value, assign the gradient to it.
                        grad_out[n, i*2+i2, j*2+j2] = grad_in[n, i, j]
        return grad_out