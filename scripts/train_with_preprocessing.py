import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Masking, TimeDistributed, Conv1D, Flatten, Reshape, Lambda, concatenate
from keras.optimizers import SGD
from keras import backend as K
from keras.engine.topology import Layer
from keras import metrics
import os
import glob

class DCTLayer(Layer):
    def __init__(self, numcep=13, norm=None, **kwargs):
        self._numcep = numcep
        self._norm = norm
        super(DCTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._N = input_shape[1]
        np_we =  np.zeros((self._N, self._N))
        for k in range(0, self._N):
            for n in range(0, self._N):
                w = np.cos((np.pi / self._N) * (n+0.5) * k)
                np_we[n, k] = w
        self.we = K.variable(np_we)
        self._trainable_weights.append(self.we)
        super(DCTLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dct_out = 2 * K.dot(x, self.we)
        dct_out = dct_out[:,:self._numcep]
        if self._norm == 'ortho':
            norm_0 = K.expand_dims(dct_out[:, 0] * K.sqrt(K.constant(1.0 / (4.0 * self._N), dtype='float32')), axis=1)
            norm_ = dct_out[:, 1:] * K.sqrt(K.constant(1.0 / (2.0 * self._N), dtype='float32'))
            dct_out = K.concatenate([norm_0, norm_], axis=1)
        return dct_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._numcep)

class FFTLayer(Layer):
    def __init__(self, NFFT=512, **kwargs):
        self._NFFT = NFFT
        super(FftLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        np_we =  np.zeros((self._NFFT, input_shape[1], 2), dtype=np.float32)
        for k in range(0, self._NFFT):
            for n in range(0, input_shape[1]):
                w = np.exp(-1j*k*n*((2*np.pi)/self._NFFT))
                np_we[k, n, 0]  = w.real
                np_we[k, n, 1] = w.imag
        self.we = K.variable(np_we.transpose(2,1,0), dtype='float32')
        self._trainable_weights.append(self.we)
        super(FftLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        sum_real = K.dot(x, self.we[0])
        sum_imag = K.dot(x, self.we[1])
        aggregated = K.stack([sum_real, sum_imag], axis=1)
        out_agg = K.sum(K.square(aggregated), axis=1)
        return out_agg[:,:257] * np.float32(1.0 / self._NFFT)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 257)

def preprocessing_model(preemph_denorm_constant=np.float32(241.25579732715335))
    inputs = Input(shape=(400, 1), name='input', dtype='float32')
    # Pre-emphasis
    x = Conv1D(128, 5, activation='relu', padding='same', name='preemph_1')(inputs)
    x = Conv1D(1, 1, padding='same', name='preemph_2')(x)
    output_preemph = Flatten()(x)
    # FFT
    x = Lambda(lambda x: x * preemph_denorm_constant, name='fft_denorm')(output_preemph)
    output_fft = FFTLayer(trainable=False, name='fft')(x)
    # Filterbanks
    x = Lambda(lambda x: K.log(x + 0.01), name='fbank_lognorm')(output_fft)
    x = Dense(512, activation='relu', name='fbank_1')(x)
    x = Dense(512, activation='relu', name='fbank_2')(x)
    output_logfbank = Dense(26, activation='linear', name='fbank_3')(x)
    # DCT
    x = DCTLayer(numcep=13, norm='ortho', trainable=False, input_shape=(26,), name='dct')(output_logfbank)
    # Lifter
    x = Dense(512, activation='relu', name='lifter_1')(x)
    x = Dense(512, activation='relu', name='lifter_2')(x)
    output_lifter = Dense(12, activation='linear', name='lifter_3')(x)
    # Log Energy
    x = Dense(1, activation='linear', input_dim=257, kernel_initializer='ones', bias_initializer='zeros',
              trainable=False, name='sum_energy')(output_fft)
    output_energy = Lambda(lambda x: K.log(x), name='log_energy')(x)
    output_mfcc = concatenate([output_energy, output_lifter])
    return inputs, output_mfcc

def rnn_model(input=None, hidden_layers=5, dropout_rate=.5, bottleneck_size=125):
    if input = None:
        input = Input(shape=(13), name='classifier_input')
    x = Masking(mask_value=0.)(input)
    for i in range(hidden_layers-1):
        x = LSTM(250, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)
    x = LSTM(bottleneck_size, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    output = TimeDistributed(Dense(39, activation='softmax'))(x)
    return input, output

