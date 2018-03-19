# ASR
Automatic Speech Recognition



#### extract_features

This standalone script is used to convert the WAV files in the TIMIT dataset to feature matrices (either MFCCs or log filterbanks). Usage:

```
python extract_features.py [-f] <source_directory> <target_directory>

<source_directory> = path/to/TIMIT/TRAIN or path/to/TIMIT/TEST
-f = extract log filterbanks instead of MFCCs
```

This script is run separately for TRAIN and TEST sets. Folders will be created in the target directory replicating the contents of the TRAIN and TEST folders. The features will be stored in .csv format. The source TIMIT contents are not modified.

The script requires [python-speech-features](https://github.com/jameslyons/python_speech_features) (tested with version 0.6), scipy (tested with version 1.0.0) and sox (tested with version 14.4.1, preinstalled on Ponyland).

TIMIT source directory is `/vol/bigdata2/smurfland_archive/databases/TIMIT` on ponies.