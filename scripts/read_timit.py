import random
import numpy as np

test_speakers = {
    "MDAB0", "MWBT0", "FELC0",
    "MTAS1", "MWEW0", "FPAS0",
    "MJMP0", "MLNT0", "FPKT0",
    "MLLL0", "MTLS0", "FJLM0",
    "MBPM0", "MKLT0", "FNLP0",
    "MCMJ0", "MJDH0", "FMGD0",
    "MGRT0", "MNJM0", "FDHC0",
    "MJLN0", "MPAM0", "FMLD0"
}

validation_speakers = {
    "FAKS0", "FDAC1", "FJEM0", "MGWT0", "MJAR0",
    "MMDB1", "MMDM2", "MPDF0", "FCMH0", "FKMS0",
    "MBDG0", "MBWM0", "MCSH0", "FADG0", "FDMS0",
    "FEDW0", "MGJF0", "MGLB0", "MRTK0", "MTAA0",
    "MTDT0", "MTHC0", "MWJG0", "FNMR0", "FREW0",
    "FSEM0", "MBNS0", "MMJR0", "MDLS0", "MDLF0",
    "MDVC0", "MERS0", "FMAH0", "FDRW0", "MRCS0",
    "MRJM4", "FCAL1", "MMWH0", "FJSJ0", "MAJC0",
    "MJSW0", "MREB0", "FGJD0", "FJMG0", "MROA0",
    "MTEB0", "MJFC0", "MRJR0", "FMML0", "MRWS1"
}


def split_development_test(test_data):
    validation_keys = []
    core_test_keys = []
    full_test_keys = []
    for key in test_data.keys():
        if key[8:10] == 'SA':
            continue
        if key[3:8] in test_speakers:
            core_test_keys.append(key)
            full_test_keys.append(key)
        elif key[3:8] in validation_speakers:
            validation_keys.append(key)
        else:
            full_test_keys.append(key)
    return validation_keys, core_test_keys, full_test_keys


class TimitGenerator(object):
    def __init__(self, datafile, keys=None, batch_size=20, shuffle=True, mask_value=0.):
        self._datafile = datafile
        self._batch_size = batch_size
        if keys is None:
            self._keys = list(datafile.keys())
        else:
            self._keys = keys
        self._batch_index = 0
        self._shuffle = shuffle
        self._mask_value = mask_value

    def shuffle_keys(self):
        random.shuffle(self._keys)

    def next_sample(self):
        key = self._keys[self._batch_index]
        self._batch_index += 1
        if self._batch_index == len(self._keys):
            self._batch_index = 0
            if self._shuffle:
                self.shuffle_keys()
        return key

    def normalize(self, sequence):
        return (sequence - np.mean(sequence, axis=0)) / np.std(sequence, axis=0)

    def pad_features(self, sequence, target_length):
        return np.pad(sequence,
                      ((0, target_length - sequence.shape[0]), (0, 0)), 'constant',
                      constant_values=((0, self._mask_value), (0, 0)))

    def pad_labels(self, sequence, target_length):
        return np.pad(sequence,
                      ((0, target_length - sequence.shape[0]), (0, 0)), 'constant',
                      constant_values=((0, 0), (0, 0)))

    def generator(self):
        while True:
            batch_keys = [self.next_sample() for _ in range(self._batch_size)]
            batch_features = [self._datafile[key]['features'] for key in batch_keys]
            batch_labels = [self._datafile[key]['labels'] for key in batch_keys]
            max_length = np.max([f.shape[0] for f in batch_features])

            sample_weights = np.zeros((self._batch_size, max_length), dtype=int)
            for i in range(self._batch_size):
                sample_weights[i, :batch_features[i].shape[0]] = 1

            batch_features = np.array([self.pad_features(self.normalize(f), max_length) for f in batch_features])
            batch_labels = np.array([self.pad_labels(f, max_length) for f in batch_labels])

            yield batch_features, batch_labels, sample_weights

