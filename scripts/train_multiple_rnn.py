#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import product
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, LSTM, Dropout, Masking, TimeDistributed
from keras.models import Sequential
from read_timit import TimitGenerator, split_development_test
import argparse
import h5py
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rnn_model(hidden_layers, dropout_rate, bottleneck_size):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, 13)))
    for i in range(hidden_layers-1):
        model.add(LSTM(250, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(bottleneck_size, return_sequences=True))
    model.add(Dropout(dropout_rate))
    # TimeDistributed is nodig om het Dense deel op iedere time step toe te passen
    model.add(TimeDistributed(Dense(39, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode='temporal')
    return model

def main(train_data_file, test_data_file):
    train_data = h5py.File(train_data_file, 'r')
    test_data = h5py.File(test_data_file, 'r')
    val_keys, core_test_keys, full_test_keys = split_development_test(test_data)

    hidden_layers = range(3, 7)
    dropout_rates = [.1, .2, .5]
    bottleneck_sizes = [125, 250, 500]

    train_generator = TimitGenerator(train_data)
    validation_generator = TimitGenerator(test_data, keys=val_keys, shuffle=False)
    for (hidden, dropout, bottleneck) in product(hidden_layers, dropout_rates, bottleneck_sizes):
        K.clear_session()
        print("Building model")
        model = rnn_model(hidden, dropout, bottleneck)
        save_path = 'weights_h{}_d{:.1f}_b{}.h5'.format(hidden, dropout, bottleneck)
        checkpointer = ModelCheckpoint(filepath=save_path, verbose=1, save_best_only=True)
        print("Start training")
        res = model.fit_generator(train_generator.generator(),
                                  steps_per_epoch=185,        # ~= 3696 (num training utterances) / 20 (batch size)
                                  epochs=20,
                                  validation_data=validation_generator.generator(),
                                  validation_steps=20,        # = 400 (num validation utterances) / 20 (batch size)
                                  callbacks=[checkpointer])
        plt.figure()
        plt.plot(res.history['acc'])
        plt.plot(res.history['val_acc'])
        plt.plot(res.history['loss'])
        plt.plot(res.history['val_loss'])
        plt.savefig('{}-training.png'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on TIMIT.")
    parser.add_argument('train_data_file', type=str, help="Train data file (.h5)")
    parser.add_argument('test_data_file', type=str, help="Test data file (.h5)")
    args = parser.parse_args()
    main(args.train_data_file, args.test_data_file)
