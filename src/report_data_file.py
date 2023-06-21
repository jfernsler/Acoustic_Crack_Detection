import argparse
import pandas as pd
import numpy as np
import glob
import os

parser = argparse.ArgumentParser(description='Acoustic Crack Detection data check - CS615.')
parser.add_argument('-file', help='Path of the file to check',
                    default='./data/acd_test_data_df.pkl', required=True)
parser.add_argument('-maketest', help='Generate test set?',
                    default=False, required=False)
parser.add_argument('-maketrain', help='Generate train set?',
                    default=False, required=False)
parser.add_argument('-maketrainsm', help='Generate smaller train set?',
                    default=False, required=False)
parser.add_argument('-ncr', help='No Crack Ratio - 50% by default',
                    default=50, required=False)
parser.add_argument('-count', help='How many samples in training set',
                    default=1000, required=False)


def count_elements(wave):
    return len(wave)

def remove_zeros(wave):
    return wave[wave!=0]

def check_data(pickle_file):
    print('\nChecking File...')
    if os.path.exists(pickle_file):
        data = pd.read_pickle(pickle_file)
        print(f'Data shape: {data.shape}')
        print(data.head(5))
        print(data.describe())

        print('\n No zero waveform info:')
        print(data['wavelen_noz'].describe())
        print()

        print(f"samples: {data.shape[0]}")
        print(f"crack samples: {data.loc[data['crack'] == 1].shape[0]}")
        print(f"no crack samples: {data.loc[data['crack'] == 0].shape[0]}")
        print()
        test3_data = data.loc[data['testname'] == 'Test3']
        test4_data = data.loc[data['testname'] == 'Test4']
        print(f"Test3 samples: {test3_data.shape[0]}")
        print(f"Test3 crack samples: {test3_data.loc[test3_data['crack'] == 1].shape[0]}")
        print(f"Test3 no crack samples: {test3_data.loc[test3_data['crack'] == 0].shape[0]}")
        print()
        print(f"Test4 samples: {test4_data.shape[0]}")
        print(f"Test4 crack samples: {test4_data.loc[test4_data['crack'] == 1].shape[0]}")
        print(f"Test4 no crack samples: {test4_data.loc[test4_data['crack'] == 0].shape[0]}")
        print()
        for i in range(1000, 7000, 1000):
            print(f"No zero waveforms with at least {i} samples: {data.loc[data['wavelen_noz'] >= i].shape[0]}")
    else:
        print('Something happened, file does not exist')

def make_test_data(pickle_file, samp_num=100):
    """
    Takes 100 samples of each the crack and non-crack data and
    makes a small batch sample data out of it
    """

    print('making sample model data')
    data = pd.read_pickle(pickle_file)

    crack_data = data.loc[data['crack'] == 1]
    nocrack_data = data.loc[data['crack'] == 0]

    # shuffle this
    crack_data = crack_data.sample(frac=1).reset_index(drop=True)
    nocrack_data = nocrack_data.sample(frac=1).reset_index(drop=True)

    nocrack_final_data = nocrack_data[:samp_num]
    crack_final_data = crack_data[:samp_num]

    model_data = crack_final_data.append(nocrack_final_data, ignore_index=True)
    print(f'final data shape: {model_data.shape}')

    model_data.to_pickle(f'./data/acd_model_data_{2*samp_num}_df.pkl')

def make_train_data(pickle_file, ncr):
    """
    Makes a full size batch of data to work with
    """
    print('making model data')
    data = pd.read_pickle(pickle_file)

    crack_data = data.loc[data['crack'] == 1]
    nocrack_data = data.loc[data['crack'] == 0]

    # 4x the crack data give 80% no crack 20% crack data
    no_crack_max = crack_data.shape[0]
    print(f'final num of no crack samples: {no_crack_max}')
    # shuffle this
    crack_data = crack_data.sample(frac=1).reset_index(drop=True)
    nocrack_data = nocrack_data.sample(frac=1).reset_index(drop=True)

    nocrack_final_data = nocrack_data[:no_crack_max]
    print(f'no crack final data shape: {nocrack_final_data.shape}')

    model_data = crack_data.append(nocrack_final_data, ignore_index=True)
    print(f'final data shape: {model_data.shape}')

    model_data.to_pickle('./data/acd_model_data_5050_df.pkl')

def make_train_data_smaller(pickle_file, count):
    """
    Makes a full size batch of data to work with
    """
    print('making model data')
    data = pd.read_pickle(pickle_file)

    crack_data = data.loc[data['crack'] == 1][:count/2]
    nocrack_data = data.loc[data['crack'] == 0]

    # 4x the crack data give 80% no crack 20% crack data
    no_crack_max = crack_data.shape[0]
    print(f'final num of no crack samples: {no_crack_max}')
    # shuffle this
    crack_data = crack_data.sample(frac=1).reset_index(drop=True)
    nocrack_data = nocrack_data.sample(frac=1).reset_index(drop=True)

    nocrack_final_data = nocrack_data[:no_crack_max]
    print(f'no crack final data shape: {nocrack_final_data.shape}')

    model_data = crack_data.append(nocrack_final_data, ignore_index=True)
    print(f'final data shape: {model_data.shape}')

    model_data.to_pickle('./data/acd_model_data_5050_sm_df.pkl')

def main():
    args = parser.parse_args()
    file_read = args.file
    make_test = args.maketest
    make_train = args.maketrain
    make_train_smaller = args.maketrainsm
    count = args.count

    nocrack_ratio = args.ncr

    check_data(file_read)

    if make_test:
        print('\nmaking test data')
        make_test_data(file_read)
    if make_train:
        print('\nmaking test data')
        make_train_data(file_read, nocrack_ratio)
    if make_train_smaller:
        print('\nmaking test data')
        make_train_data_smaller(file_read, count)


if __name__ == '__main__':
    main()
