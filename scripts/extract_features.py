from python_speech_features import mfcc, logfbank
from python_speech_features.sigproc import framesig
import scipy.io.wavfile as wav
import argparse as ap
import sys
import glob
import os
import numpy as np
import subprocess
import h5py

TEST_NAMES = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey',
              'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p',
              'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z']

FOLDINGS = {
    "ux": "uw",
    "axr": "er",
    "axh": "ah",
    "ax-h": "ah",
    "ax": "ah",
    "em": "m",
    "nx": "n",
    "en": "n",
    "eng": "ng",
    "hv": "hh",
    "el": "l",
    "zh": "sh",
    "ao": "aa",
    "ix": "ih",
    "pcl": "sil",
    "tcl": "sil",
    "kcl": "sil",
    "qcl": "sil",
    "bcl": "sil",
    "dcl": "sil",
    "gcl": "sil",
    "h#": "sil",
    "#h": "sil",
    "pau": "sil",
    "epi": "sil"
}


def phn_file_to_labels(fname, winstep=0.01, rate=16000.0):
    # Load PHN file
    phn_text = np.loadtxt(fname,
                          dtype={'names': ('start', 'end', 'phone'),
                                 'formats': (np.int32, np.int32, 'S4')},
                          comments=None)

    # Length of annotations in num samples
    _, phn_len, _ = phn_text[-1]
    labels = []
    for i, (start, end, phn) in enumerate(phn_text):
        # Round the start & end times to the window start
        start = int(start / (rate * winstep))
        end = int(end / (rate * winstep))
        labels.extend([phn.decode("utf-8")] * (end - start))

    return np.asarray(labels)


def fold_label(phn):
    if phn in FOLDINGS:
        return FOLDINGS[phn]
    else:
        return phn


def label_to_int(phn):
    return TEST_NAMES.index(phn)


def to_one_hot(labels, num_classes=48):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot


def main(source_dir, target_dir,
         use_logfbank=False,
         winstep=0.01,
         winlen=0.025,
         include_sa=False,
         pad_labels=False,
         raw_audio=False):
    drs = glob.glob(os.path.join(source_dir, 'DR*'))
    if len(drs) == 0:
        raise ValueError("Source folder does not contain DR# subfolders")

    cmd = "sox {0} -t wav {1}"

    target_filename = os.path.join(target_dir, 'features.h5')

    target_file = h5py.File(target_filename, "w")


    print("Start tree walk")
    for dialect_region_dir in drs:
        print(dialect_region_dir)
        dr = os.path.split(dialect_region_dir)[-1]
        for speaker_dir in glob.glob(os.path.join(dialect_region_dir, "*")):
            speaker = os.path.split(speaker_dir)[-1]
            print(speaker_dir)
            for audio_file in glob.glob(os.path.join(speaker_dir, "*.WAV")):
                fname = os.path.splitext(os.path.split(audio_file)[-1])[0]
                if not include_sa and audio_file[8:10] == "SA":
                    continue
                print(audio_file)
                file_target_dir = os.path.join(target_dir, dr, speaker)
                os.makedirs(file_target_dir, exist_ok=True)

                # Convert NIST wav files using sox
                converted_target_file = os.path.join(file_target_dir, os.path.split(audio_file)[-1])
                if not os.path.isfile(converted_target_file):
                    subprocess.call(cmd.format(audio_file, converted_target_file), shell=True)


                (rate, sig) = wav.read(converted_target_file)
                if use_logfbank:
                    features = logfbank(sig, rate, winlen=winlen, winstep=winstep)
                else:
                    if raw_audio:
                        features = framesig(sig, winlen*rate, winstep*rate)
                    else:
                        features = mfcc(sig, rate, winlen=winlen, winstep=winstep)


                # Get the phonetic labels (no foldings, q phones included)
                labels = phn_file_to_labels(os.path.join(source_dir, dr, speaker, fname + '.PHN'),
                                            winstep)

                # If labels extend beyond speech signal, always crop
                if labels.shape[0] > features.shape[0]:
                    labels = labels[:features.shape[0]]

                # If speech signal extends beyond labels, crop signal or pad labels
                if labels.shape[0] < features.shape[0]:
                    if pad_labels:
                        labels = np.append(labels, [labels[-1]] * (features.shape[0] - labels.shape[0]))
                    else:
                        features = features[:labels.shape[0]]
                assert features.shape[0] == labels.shape[0], "Something went wrong padding or cropping."

                # Remove 'q' phones
                mask = np.ones(features.shape[0], dtype=bool)
                q_idxs = np.where(labels == 'q')[0]
                mask[q_idxs] = False
                features = features[mask]
                labels = labels[mask]
                # Fold labels
                labels = [fold_label(phn) for phn in labels]
                # Convert to one-hot
                labels = np.asarray([label_to_int(phn) for phn in labels])
                labels = to_one_hot(labels, 39)

                group = target_file.create_group(fname)
                group.create_dataset('features', data=features)
                group.create_dataset('labels', data=labels)
    target_file.close()


if __name__ == '__main__':
    if len(sys.argv) > 0:
        parser = ap.ArgumentParser(description='Convert TIMIT wav files to feature vectors.')
        parser.add_argument('source_directory', type=str, help='location of TRAIN or TEST folder')
        parser.add_argument('target_directory', type=str, help='target directory for feature vectors')
        parser.add_argument('-f', dest='use_fb', action='store_true',
                            help='extract log filterbank energies instead of MFCC vectors')
        parser.add_argument('--raw', dest='raw_audio', action='store_true',
                            help='extract raw audio in frames instead of MFCC vectors')
        parser.add_argument('--winlen', dest='winlen', type=float, default=0.025,
                            help='window length for feature extraction in seconds (default 0.025)')
        parser.add_argument('--winstep', dest='winstep', type=float, default=0.01,
                            help='window step for feature extraction in seconds (default 0.01)')
        parser.add_argument('--sa', dest='include_sa', action='store_true',
                            help="Include SA sentences")
        parser.add_argument('--pad_labels', dest='pad_labels', action='store_true',
                            help="Repeat the last label (always silence) until the end of the utterance, " +
                                 "default behavior deletes the feature vectors after the end of annotations.")
        args = parser.parse_args()
        main(args.source_directory, args.target_directory, args.use_fb, args.winstep, args.winlen,
             args.include_sa, args.pad_labels, args.raw_audio)
