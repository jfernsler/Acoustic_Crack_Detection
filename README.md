# Acoustic Crack Detection
Deep Learning Approach for Acoustic Crack Detection

This is a project was created as an excersise in scratch building deep learning layers using only python and numpy to gain a deeper understanding of the mathematics and structure behind the process.

Acoustic waveforms are converted into spectrograms and fed into our convolutional neural network to analyze for moments of stress fractures while the sample is under strain.

## Requires
* pandas 1.0.1
* matplotlib 3.5.1
* numpy 1.22.2
* scipy 1.8.0
  * scipy.fft (for spectrogram)
  * scipy.stats (for entropy)

## New data shape

The dataframe now contains the following columns:
* channel : data channel (should always be 3)
* crack : is this a crack sample (0,1)
* hit : which hit number
* test_name : which test set it came from (currently only Test4)
* time : Time
* waveform : The waveform array
* waveform_noz : The waveform array with no zero data
* wavelen_noz : The length of the no-zero waveform

We can use the wavelen_noz to ensure we're only using waveforms with consistent amounts of data

## Instructions for forming data:
1. process large data file using construct_data_file.py:

```
construct_data_file.py --help
usage: construct_data_file.py [-h] [-path PATH] [-write WRITE] [-file FILE]

Acoustic Crack Detection data prep - CS615.

optional arguments:
  -h, --help    show this help message and exit
  -path PATH    Path to the data files
  -write WRITE  Path to write the pickle, if big data, keep it out of the git repo
  -file FILE    Name of the file
```

2. check that new data file and generate test data using report_data_file.py:

```
python3 report_data_file.py --help
usage: report_data_file.py [-h] -file FILE [-maketest MAKETEST]

Acoustic Crack Detection data check - CS615.

optional arguments:
 -h, --help          show this help message and exit
 -file FILE          Path of the file to check
 -maketest MAKETEST  Generate test set?
  ```

for example, I am keeping my data outside of the repo at:
 ``` ../../acd_data/acd_data_df.pkl ```

Therefore to check it and create the small sample data in the repo:

```report_data_file.py -file ../../acd_data/acd_data_df.pkl -maketest True```

which will place 200 samples of data in the data folder
