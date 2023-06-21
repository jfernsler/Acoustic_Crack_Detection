from tokenize import Special
import numpy as np
import pandas as pd
from scipy import fft
from scipy import stats
from PIL import Image
import matplotlib.pyplot as plt
#from skimage.transform import resize

from dnn.base_classes import Layer

# global epsilon
epsilon = 0.0000001

# Input Layer ##
class InputWaveformLayer(Layer):
    """
    A class to represent an InputLayer for an acoustic waveform.
    """
    def __init__(self, X,
                    imtype='spectrogram',
                    size=128,
                    resize_type='nearest',
                    samp_size=3000,
                    fft_window=77,):
        """
        Constructor

        Parameters
        ----------
            X : array
                Standard deviation of the input array
            size : 2 value tuple
                final size of the spectrogram
            resize_type : string
                nearest -> using nearest neighbor to resize the spectrogram
                bilinear -> smoother resizing
            samp_size : int
                how many samples of the waveform to use starting at index zero
            fft_window: int
                how many samples to use for frequency detection
        """
        super().__init__()
        self.setPrevIn(X)
        self.__imtype = imtype
        self.__spec_size = (size, size)
        self.__resize_type = resize_type
        self.__sample_size = samp_size
        self.__fft_window = fft_window
        self.__entropy = np.array([])
        self.calculate_entropy(X)


    def calculate_entropy(self, X):
        """
        Calculate shanon entropy for all input
        """
        for w in X:
            pd_series = pd.Series(w)
            counts = pd_series.value_counts()
            self.__entropy = np.append(self.__entropy, stats.entropy(counts))


    def get_entropy(self):
        """
        Return entropy values for waveform
        """
        return self.__entropy


    def make_spectrogram(self, waveform, N, rate=5000000):
        """
        Create and return a spectrogram array
        """
        # Where to put the spectrogram
        S = []
        for k in range(0, waveform.shape[0]+1, N):
            x = fft.fftshift(fft.fft(waveform[k:k+N], n=N))[N//2:N]
            Pxx = 10*np.log10(np.real(x*np.conj(x)) + epsilon)
            S.append(Pxx)
        S = np.array(S)

        # Calc Frequencies - don't return those yet
        #f = fft.fftshift(fft.fftfreq(N, d=1/rate))[N//2:N]
        return S.T


    def normalize_data(self, data):
        """
        Normalize an array 0-1
        """
        return (data - np.min(data)) / (np.max(data) - np.min(data) + epsilon)


    def resize_data(self, data):
        """
        Resize data using to a consistent size
        """
        img = Image.fromarray(data, 'F')
        if self.__resize_type == 'nearest':
            r_img = img.resize(self.__spec_size, Image.NEAREST)
        elif self.__resize_type == 'bilinear':
            r_img = img.resize(self.__spec_size, Image.BILINEAR)
        else:
            r_img = img.resize(self.__spec_size, Image.NEAREST)
        return np.asarray(r_img)

    def generate_spectrogram_image(self, waveform):
        """
        Make a spectrogram, normalize it, resize it, return it
        """
        spec = self.make_spectrogram(waveform[:self.__sample_size], N=self.__fft_window)
        n_spec = self.normalize_data(spec)
        r_spec = self.resize_data(n_spec)
        return r_spec


    def canvas2rgb_array(self, canvas):
        """Adapted from: https://stackoverflow.com/a/21940031/959926"""
        canvas.draw()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = canvas.get_width_height()
        scale = round((buf.size / 3 / nrows / ncols)**.5)
        return buf.reshape(scale * nrows, scale * ncols, 3)

    def generate_waveform_image(self, waveform):
        # init figure
        fig, ax = plt.subplots()
        # set plot limits (data says these should work)
        plt.ylim(-.03, .03)
        plt.xlim(0,self.__sample_size)
        # turn off axis
        plt.axis('off')

        # these are a bunch of 'magic' settings that ensure
        # no borders, axis, or other extraneous stuff
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.margins(0)

        # finally create the plot
        ax.plot(waveform)
        plt.close()

        # Extract the plot as an array
        plt_array = self.canvas2rgb_array(fig.canvas)
        # get the first channel, normalize it
        wave_chan01 = self.normalize_data(plt_array[...,0])
        # resize it using skimage, PIL does not play well with this.
        #wave_resize= resize(wave_chan01, (256, 256))
        wave_resize= self.resize_data(wave_chan01)
        return wave_resize


    def forward(self, X):
        """
        Generates the spectrogram of the input array
        """
        image_tensor = []
        self.setPrevIn(X)
        for w in X:
            if self.__imtype == 'waveform':
                image = self.generate_waveform_image(w)
            else:
                image = self.generate_spectrogram_image(w)
            image_tensor.append(image)
        out_tensor = np.stack(image_tensor)
        self.setPrevOut(out_tensor)
        return out_tensor

    def gradient(self):
        """Unimplemented"""
        pass
