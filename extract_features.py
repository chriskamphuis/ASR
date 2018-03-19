from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import argparse as ap
import sys
import glob
import os
import numpy as np
import subprocess


def main(source_dir, target_dir, use_logfbank=False):
    drs = glob.glob(os.path.join(source_dir, 'DR*'))
    if len(drs) == 0:
        raise ValueError("Source folder does not contain DR# subfolders")

    cmd = "sox {0} -t wav {1}"
    mv_cmd = "mv {0} {1}"
    
    print("Start tree walk")
    for dialect_region_dir in drs:
        print(dialect_region_dir)
        dr = os.path.split(dialect_region_dir)[-1]
        for speaker_dir in glob.glob(os.path.join(dialect_region_dir, "*")):
            speaker = os.path.split(speaker_dir)[-1]
            print(speaker_dir)
            for audio_file in glob.glob(os.path.join(speaker_dir, "*.WAV")):
                print(audio_file)
                file_target_dir = os.path.join(target_dir, dr, speaker)
                os.makedirs(file_target_dir, exist_ok=True)
                
                # Convert NIST wav files using sox
                converted_target_file = os.path.join(file_target_dir, os.path.split(audio_file)[-1])
                if not os.path.isfile(converted_target_file):
                    subprocess.call(cmd.format(audio_file, converted_target_file), shell=True)
                
                fname = os.path.splitext(os.path.split(audio_file)[-1])[0]
                (rate, sig) = wav.read(converted_target_file)
                if use_logfbank:
                    features = logfbank(sig, rate)
                else:
                    features = mfcc(sig, rate)

                np.savetxt(os.path.join(file_target_dir, fname+'.csv'), features, delimiter=',')


if __name__ == '__main__':
    if len(sys.argv) > 0:
        parser = ap.ArgumentParser(description='Convert TIMIT wav files to feature vectors.')
        parser.add_argument('source_directory', type=str, help='location of TRAIN or TEST folder.')
        parser.add_argument('target_directory', type=str, help='target directory for feature vectors')
        parser.add_argument('-f', dest='use_fb', action='store_true',
                            help='extract log filterbank energies instead of MFCC vectors')
        args = parser.parse_args()
        main(args.source_directory, args.target_directory, args.use_fb)
