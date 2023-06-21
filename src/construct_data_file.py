import argparse
import pandas as pd
import numpy as np
import glob
import os

parser = argparse.ArgumentParser(description='Acoustic Crack Detection data prep - CS615.')
parser.add_argument('-path', help='Path to the data files',
                    default='../snippets/test_data/', required=False)
parser.add_argument('-write', help='Path to write the pickle, if big data, keep it out of the git repo',
                    default='../../acd_data/', required=False)
parser.add_argument('-file', help='Name of the file',
                    default='acd_test_data_df.pkl', required=False)

### Helper functions for pandas
def count_elements(wave):
    return len(wave)

def remove_zeros(wave):
    return wave[wave!=0]

### rest of the functions
def read_data_file(f_name):
    """
    Read in a data file from coupon testing, return a dictionary of values.
    """
    data_dict = {}
    file = open(f_name, "r")

    test_name = file.name.split('/')[-1]
    test_name = test_name.split('_')[0]

    # grab some extra info
    all_data = file.readlines()
    data_dict['channel'] = int(all_data[8].split(' ')[-1])
    data_dict['hit'] = int(all_data[9].split(' ')[-1])
    data_dict['time'] = float(all_data[10].strip('\n').split(' ')[-1])
    data_dict['testname'] = test_name

    # get the waveform
    wave = all_data[13:]
    data_array = np.array(wave).astype('float32')
    data_dict['waveform'] = data_array

    return data_dict


def main(path_read, path_write, file_write):
    # using glob now to grab wildcard txt files
    file_list = glob.glob(path_read + '*.txt')

    # set up our data columns
    data_columns = ['testname', 'channel', 'hit', 'time', 'waveform', 'crack']
    data = pd.DataFrame()
    print(data)

    # set the crack time
    #crack_time = 325.6337890625
    crack_times = {}
    crack_times['Test3'] = 314.4658203125
    crack_times['Test4'] = 325.6337890625
    channel = {}
    channel['Test3'] = 3
    channel['Test4'] = 1

    print(f'\n\nWill save to {path_write+file_write}')
    print(f'Reading {len(file_list)} files from {path_read}\n')
    # read them all in
    for n,file in enumerate(file_list):
        try:
            data_in = read_data_file(file)
            if n%100 == 0:
                print(f"{data_in['testname']}:{n} ", end='', flush=True)
        except:
            print(f'\n\nError on : {file}\n')

        if data_in['time'] >= crack_times[data_in['testname']]:
            data_in['crack'] = 1
        else:
            data_in['crack'] = 0

        #print(data_in)

        # we are only interested in one channel
        if data_in['channel'] == channel[data_in['testname']]:
            data = data.append(data_in, ignore_index=True)
            # create a column of waveform with no zeros
            data['waveform_noz'] = data['waveform'].apply(remove_zeros)
            # count up those data points and create a new column
            data['wavelen_noz'] = data['waveform_noz'].apply(count_elements)

    # set all values after a specific time to be crack values
    #data.loc[data['time'] >= crack_times[data['testname']], 'crack'] = 1

    # put it in a time series
    data.sort_values(['time'], ignore_index=True, inplace=True)

    # check
    print(data.head())
    print(data.shape)

    # Check whether the specified path exists or not
    if not os.path.exists(path_write):
        # Create a new directory because it does not exist
        os.makedirs(path_write)
        print(f'Created directory at: {path_write}')

    # write it out
    try:
        data.to_pickle(path_write + file_write)
        print(f'Dataframe successfully pickled to {path_write + file_write}')
    except:
        print('Failed to picke the Dataframe')


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


if __name__ == '__main__':
    args = parser.parse_args()
    path_read = args.path
    path_write = args.write
    file_write = args.file

    if file_write[-4:] != '.pkl':
        file_write += '.pkl'

    main(path_read, path_write, file_write)
    check_data(path_write + file_write)
