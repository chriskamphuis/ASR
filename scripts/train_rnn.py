import h5py
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Masking, TimeDistributed
from keras.callbacks import ModelCheckpoint
from read_timit import TimitGenerator, split_development_test
import argparse
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt


def rnn_model():
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(None, 13)))
    model.add(LSTM(250, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(250, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(250, return_sequences=True))
    model.add(Dropout(0.5))
    # TimeDistributed is nodig om het Dense deel op iedere time step toe te passen
    model.add(TimeDistributed(Dense(39, activation='softmax')))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], sample_weight_mode='temporal')
    return model

def main(train_data_file, test_data_file, weights_file):
    train_data = h5py.File(train_data_file, 'r')
    test_data = h5py.File(test_data_file, 'r')
    val_keys, core_test_keys, full_test_keys = split_development_test(test_data)

    train_generator = TimitGenerator(train_data)
    validation_generator = TimitGenerator(test_data, keys=val_keys, shuffle=False)
    print("Building model")
    model = rnn_model()
    print("Done")
    checkpointer = ModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=True)
    print("Start training")
    model.fit_generator(train_generator.generator(),
                        steps_per_epoch=185,        # ~= 3696 (num training utterances) / 20 (batch size)
                        epochs=20,
                        validation_data=validation_generator.generator(),
                        validation_steps=20,        # = 400 (num validation utterances) / 20 (batch size)
                        callbacks=[checkpointer])




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on TIMIT.")
    parser.add_argument('train_data_file', type=str, help="Train data file (.h5)")
    parser.add_argument('test_data_file', type=str, help="Test data file (.h5)")
    parser.add_argument('weights_file', type=str, help="Weights data file (.hdf5)")
    args = parser.parse_args()
    main(args.train_data_file, args.test_data_file, args.weights_file)
