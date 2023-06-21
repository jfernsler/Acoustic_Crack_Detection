import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
import os
from math import floor

from dnn import InputWaveformLayer
from dnn import ConvolutionalLayer
from dnn import ReLuLayer
from dnn import FlattenLayer
from dnn import FullyConnected
from dnn import SoftmaxLayer
from dnn import CrossEntropy
from dnn import NeuralNetwork
from dnn import SigmoidLayer
from dnn import LogLoss
from dnn import encode_onehot, decode_onehot
from dnn import TanhLayer


parser = argparse.ArgumentParser(description='Acoustic Crack Detection - CS615.')
# parser.add_argument('-maxepoch', help='maximum epoch count',
#                      default='200', required=False)
# parser.add_argument('-eta', help='set learning rate',
#                      default='0.0001', required=False)
# parser.add_argument('-batch', help='batch size for data',
#                      default='50', required=False)
# parser.add_argument('-thresh', help='early stopping threshold 10^-[thresh]',
#                      default='8', required=False)
# parser.add_argument('-init', help='Weights and bias parameter initiation method',
#                      choices=['default', 'xavier', 'he'], default='default', required=False)
parser.add_argument('-test', help='When testing smaller things...',
                    default='ms3', required=False)
parser.add_argument('-data', help='Name of the file',
                    default='./data/acd_model_data_5050_sm_df.pkl', required=False)


def split_train_valid(data, train_ratio=0.8):
    """
    Split the ACD data set into training and validation data

    Returns both train and validation data
    """
    crack_data = data.loc[data['crack'] == 1]
    nocrack_data = data.loc[data['crack'] == 0]

    crack_count = crack_data.shape[0]
    nocrack_count = nocrack_data.shape[0]

    # for 80train 20valid:
    train_crack_count = floor(crack_count * train_ratio)
    train_nocrack_count = floor(nocrack_count * train_ratio)
    # split it out for crack data
    train_crack = crack_data[:train_crack_count]
    valid_crack = crack_data[train_crack_count:]
    # split it out for no-crack data
    train_nocrack = nocrack_data[:train_nocrack_count]
    valid_nocrack = nocrack_data[train_nocrack_count:]
    # merge training and validation data_in
    train_data = train_crack.append(train_nocrack, ignore_index=True)
    valid_data = valid_crack.append(valid_nocrack, ignore_index=True)
    # shuffle it up
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    valid_data = valid_data.sample(frac=1).reset_index(drop=True)

    return train_data, valid_data



def test_01(data_file):
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return

    print(data.shape)
    print(data.describe)


    # set some parameters
    # 3000 samples and a window of 77 makes a nice square spectrogram
    # but with the resize it doesn't really matter
    sample_count = 3000
    img_size = 128
    fft_win = 77

    train_data, valid_data = split_train_valid(data)
    # select out the rows that have at set number of
    # non-zero data samples
    train_data = train_data.loc[train_data['wavelen_noz'] >= sample_count]
    X = train_data['waveform_noz']

    L0 = InputWaveformLayer(X, size=img_size, samp_size=sample_count, fft_window=fft_win, resize_type='bilinear')
    L1 = ConvolutionalLayer(1, 128, 128, 5, 32)
    L2 = ReLuLayer()
    L3 = FlattenLayer()

    f, axarr = plt.subplots(3)

    h = X
    print('L0 (Input)-> forward')
    h = L0.forward(h)
    axarr[0].set_title('Input')
    axarr[0].imshow(h[1])

    print('L1 (Conv)-> forward')
    h = L1.forward(h)
    axarr[1].set_title('Convolutional')
    axarr[1].imshow(h[1])

    print('L2 (ReLu)-> forward')
    h = L2.forward(h)
    axarr[2].set_title('ReLu')
    axarr[2].imshow(h[1])
    plt.show()

    print('L3 (Flatten)-> forward')
    h = L3.forward(h)

    print(f'{type(L3).__name__} out shape: {h.shape}')
    print(f'{type(L0).__name__} entropy shape: {L0.get_entropy().shape}')

    plt.plot(L0.get_entropy())
    plt.ylabel('entropy')
    plt.xlabel('observation')
    plt.title('Shannon Entropy')
    plt.show()

    return


def test_02(data_file):
    print('\nrunning test2...')

    # bring in the data
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return

    print(data.shape)
    print(data.describe)

    # select out the rows that have at set number of
    # non-zero data samples
    sample_count = 3000

    train_data, valid_data = split_train_valid(data)
    # select out the rows that have at set number of
    # non-zero data samples
    train_data = train_data.loc[train_data['wavelen_noz'] >= sample_count]
    X = train_data['waveform_noz']
    # set input layer data
    # 3000 samples and a window of 77 makes a nice square spectrogram
    L0 = InputWaveformLayer(X, size=256, samp_size=sample_count)

    print('L0 (Input)-> forward')
    h = X
    h = L0.forward(h)

    print(f'{type(L0).__name__} out shape: {h.shape}')
    print(f'{type(L0).__name__} entropy shape: {L0.get_entropy().shape}')

    f, axarr = plt.subplots(2)
    axarr[0].set_title('Input')
    axarr[0].imshow(h[1])
    axarr[1].plot(L0.get_entropy())
    plt.show()
    return

def test_ms3(data_file):
    print('\n#### Start test_ms3 ####\n')
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return


    # set some parameters
    # 3000 samples and a window of 77 makes a nice square spectrogram
    # but with the resize it doesn't really matter
    sample_count = 3000
    img_size = 128
    img_ch = 1
    fft_win = 77
    kernel_size = 3
    conv_out1 = 32
    num_classes=2 #0 no crack, 1 crack
    fc_in_count = conv_out1*(img_size-(kernel_size-1))**2

    # get the data
    train_data, valid_data = split_train_valid(data)

    # select out the rows that have at set number of
    # non-zero data samples for training
    train_data = train_data.loc[train_data['wavelen_noz'] >= sample_count]
    x_train = train_data['waveform_noz']
    y_train = encode_onehot(num_classes, train_data['crack'].to_numpy().reshape(-1,1))
    y_train = y_train.reshape(len(y_train), 1, num_classes)

    # select out validation data
    valid_data = valid_data.loc[valid_data['wavelen_noz'] >= sample_count]
    x_valid = valid_data['waveform_noz']
    y_valid = encode_onehot(num_classes, valid_data['crack'].to_numpy().reshape(-1,1))
    y_valid = y_valid.reshape(len(y_valid), 1, num_classes)

    layers = [
        InputWaveformLayer(x_train, size=img_size, samp_size=sample_count, fft_window=fft_win, resize_type='bilinear'),
        ConvolutionalLayer(img_ch, img_size, img_size, kernel_size, conv_out1),
        TanhLayer(),
        ReLuLayer(),
        FlattenLayer(),
        FullyConnected(fc_in_count, out_count=2),
        TanhLayer(),
        SoftmaxLayer()
    ]

    net = NeuralNetwork('test_model', layers, objective=CrossEntropy(), train_type='sgd', optimizer='adam')

    net.train(x_train, y_train,
              epochs=100, batch_size=40,
              eta=0.002, criteria=10**-2, random_seed=0,
              x_valid=x_valid, y_valid=y_valid,
              )
    return


def test_ms4(data_file):
    print('\n#### Start test_ms4 ####\n')
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return


    # set some parameters
    # 3000 samples and a window of 77 makes a nice square spectrogram
    # but with the resize it doesn't really matter
    sample_count = 3000
    img_size = 128
    img_ch = 1
    fft_win = 77
    kernel_size = 5
    conv_out1 = 32
    conv_out2 = 16
    num_classes=2 #0 no crack, 1 crack
    fc_in_count = conv_out2*(img_size-(kernel_size-1)-(kernel_size-1))**2

    # get the data
    train_data, valid_data = split_train_valid(data)

    # select out the rows that have at set number of
    # non-zero data samples for training
    train_data = train_data.loc[train_data['wavelen_noz'] >= sample_count]
    x_train = train_data['waveform_noz']
    y_train = encode_onehot(num_classes, train_data['crack'].to_numpy().reshape(-1,1))
    y_train = y_train.reshape(len(y_train), 1, num_classes)

    # select out validation data
    valid_data = valid_data.loc[valid_data['wavelen_noz'] >= sample_count]
    x_valid = valid_data['waveform_noz']
    y_valid = encode_onehot(num_classes, valid_data['crack'].to_numpy().reshape(-1,1))
    y_valid = y_valid.reshape(len(y_valid), 1, num_classes)

    layers = [
        InputWaveformLayer(x_train, size=img_size, samp_size=sample_count, fft_window=fft_win, resize_type='bilinear'),
        ConvolutionalLayer(img_ch, img_size, img_size, kernel_size, conv_out1),
        TanhLayer(),
        ReLuLayer(),
        ConvolutionalLayer(conv_out1, img_size-(kernel_size-1), img_size-(kernel_size-1), kernel_size, conv_out2),
        TanhLayer(),
        ReLuLayer(),
        FlattenLayer(),
        FullyConnected(fc_in_count, out_count=2),
        TanhLayer(),
        SoftmaxLayer()
    ]

    net = NeuralNetwork('test_model', layers, objective=CrossEntropy(), train_type='sgd', optimizer='adam')

    net.train(x_train, y_train,
            epochs=50, batch_size=40,
            eta=0.001, criteria=10**-3, random_seed=0,
            x_valid=x_valid, y_valid=y_valid,
            )

    return



def test_jf4(data_file):
    print('\nrunning testjf4...')
    print('splitting out training and validation data on an 80/20 split')
    # bring in the data
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return

    print(f'\nIncoming data size: {data.shape[0]} rows\n')

    train_data, valid_data = split_train_valid(data)

    # check it...
    tsize = train_data.shape[0]
    vsize = valid_data.shape[0]

    print(f'train size: {tsize} : {tsize/data.shape[0] * 100}%')
    print(f'valid size: {vsize} : {vsize/data.shape[0] * 100}%')
    print()
    return

def test_jf5(data_file):
    """
    Testing injection of addtional data to the fully connected layer
    """
    print('\nrunning testjf5...')
    X1 = np.array([1,2,3,4,5,6])[...,None]
    X2 = np.array([[2,3],[3,4],[5,6],[7,8],[8,9],[9,10]])
    outputs = 1
    L1 = FullyConnected(X1.shape[1], outputs, additional_features = 2)
    print(f'weights shape {L1.getWeights().shape}')

    L1.set_additional_features(X2)
    h = L1.forward(X1)
    print(h.shape)
    print()
    print('normal')
    L2 = FullyConnected(X1.shape[1], outputs)
    print(f'weights shape {L1.getWeights().shape}')
    h = L2.forward(X1)
    print(h.shape)

    return


def test_jf6(data_file):
    print('\n#### Start test_jf6 ####\n')
    print('\nChecking File...')
    if os.path.exists(data_file):
        data = pd.read_pickle(data_file)
    else:
        print(f'no file at: {data_file}')
        return


    # set some parameters
    # 3000 samples and a window of 77 makes a nice square spectrogram
    # but with the resize it doesn't really matter
    sample_count = 3000
    img_size = 128
    img_ch = 1
    fft_win = 77
    kernel_size = 5
    conv_out1 = 32
    num_classes=2 #0 no crack, 1 crack
    fc_in_count = conv_out1*(img_size-(kernel_size-1))**2

    # get the data
    train_data, valid_data = split_train_valid(data)

    # select out the rows that have at set number of
    # non-zero data samples for training
    train_data = train_data.loc[train_data['wavelen_noz'] >= sample_count]
    x_train = train_data['waveform_noz']
    y_train = encode_onehot(num_classes, train_data['crack'].to_numpy().reshape(-1,1))
    y_train = y_train.reshape(len(y_train), 1, num_classes)

    # select out validation data
    valid_data = valid_data.loc[valid_data['wavelen_noz'] >= sample_count]
    x_valid = valid_data['waveform_noz']
    y_valid = encode_onehot(num_classes, valid_data['crack'].to_numpy().reshape(-1,1))
    y_valid = y_valid.reshape(len(y_valid), 1, num_classes)


    layers = [
        InputWaveformLayer(x_train, size=img_size, samp_size=sample_count, fft_window=fft_win, resize_type='bilinear'),
        ConvolutionalLayer(img_ch, img_size, img_size, kernel_size, conv_out1),
        TanhLayer(),
        ReLuLayer(),
        FlattenLayer(),
        FullyConnected(fc_in_count, out_count=2),
        TanhLayer(),
        SoftmaxLayer()
    ]

    net = NeuralNetwork('test_model', layers, objective=CrossEntropy(), train_type='sgd', optimizer='adam')

    net.train(x_train, y_train, 
                epochs=50, batch_size=4, 
                eta=0.001, criteria=10**-3, random_seed=0,
                x_valid=x_valid, y_valid=y_valid,
                )

    return

if __name__ == '__main__':
    # parse the args...
    args = parser.parse_args()
    # eta = float(args.eta)
    # maxepoch = int(args.maxepoch)
    # batch = int(args.batch)
    # thresh = int(args.thresh)
    # wb_init = args.init
    test = args.test
    data_file = args.data

    if test in ['1']:
        test_01(data_file)
        print('\n\n')
    if test in ['2']:
        test_02(data_file)
        print('\n\n')
    if test in ['ms3']:
        test_ms3(data_file)
        print('\n\n')
    if test in ['ms4']:
        test_ms4(data_file)
        print('\n\n')
    if test in ['jf4']:
        test_jf4(data_file)
        print('\n\n')
    if test in ['jf5']:
        test_jf5(data_file)
        print('\n\n')
    if test in ['jf6']:
        test_jf6(data_file)
        print('\n\n')
