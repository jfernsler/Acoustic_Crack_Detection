import numpy as np
import os
import pandas as pd
from numpy.lib import stride_tricks
from numpy.lib.stride_tricks import as_strided
# from numpy.lib.stride_tricks import sliding_window_view
import skimage.measure
from skimage.util import view_as_blocks


class MaxPoolLayer():
    def __init__(self):
        super().__init__()

    def forward(self, X, window_size):
        print('original')
        print(X)
        pad_w = []
        for i in range(len(window_size)):
            if X.shape[i] % window_size[i] != 0:
                after_width = window_size[i] - (X.shape[i] % window_size[i])
            else:
                after_width = 0
            pad_w.append((0, after_width))
        image_withpad = np.pad(X, pad_width=pad_w, mode='constant')
        window = view_as_blocks(image_withpad, window_size)
        self.output = np.max(window, axis=tuple(range(image_withpad.ndim, window.ndim)))
        return self.output

    def ideal_forward(self, X):
        # Ideal Output
        ideal_output = skimage.measure.block_reduce(X, (3, 2, 2), np.max)
        return ideal_output

    def backwawrd(self, X, gradIn):
        
        return gradIn



if __name__ == '__main__':
    a = np.array([[13, 45, 67, 4], [1, 2, 3, 4], [2, 3, 4, 6], [1, 23, 44, 1]])
    b = np.array([[11, 22, 33, 57], [1, 2, 3, 4], [2, 3, 94, 6], [1, 23, 44, 1]])
    c = np.array([[1, 2, 3, 4], [1, 7, 8, 9], [2, 3, 4, 76], [1, 23, 44, 1]])
    input = np.stack((a, b, c))
    print(np.shape(input))

    maxp = MaxPoolLayer()
    print('*******')
    out = maxp.forward(input, window_size=(3, 2, 2))
    print(out)
    # maxp.window_stack(res, stepsize=2, width=3)
    # print('*******')
    # result = maxp.maxpool_multiple(input)
    # print(result)
    # print('*******')
    # result2= maxp.asStride(input, (1, 2, 2), 2)
    # print(result2)
    ideal = maxp.ideal_forward(input)
    print(f'ideal output:\n {ideal}')
    print('*******')

    gradIn = np.array([[1,2], [3,4]])
    backprop = maxp.backwawrd(input, gradIn)
    print(f'BackProp Result: \n {backprop}')

    # maxp2 = MaxPoolLayerSM()
    # myresult = maxp2.forward_pass(input)
    # print(myresult)