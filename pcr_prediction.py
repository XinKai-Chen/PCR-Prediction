import numpy as np
from scipy import interpolate
import pylab as pl
import pandas as pd
import os
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout

import sys
sys.path.append('../../')
import configs.config as cfg

cur_dir_path = os.getcwd()


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))),
                      (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}
        self.saveLoss = []
        self.savevalLoss = []

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.saveLoss.append(logs.get('loss'))
        self.saveLoss.append(logs.get('loss'))

    # self.savevalLoss.append(logs.get('val_loss'))
    # self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        self.saveLoss.append(logs.get('loss'))
        self.savevalLoss.append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.show()


class TextAttBiRNN(object):
    def __init__(self, maxlen, max_features, embedding_dims,
                 class_num=2,
                 last_activation='softmax'):
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen,))

        embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        x = Bidirectional(LSTM(512, return_sequences=True))(embedding)  # LSTM or GRU
        x = Attention(self.maxlen)(x)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)
        return model


def getRawdata(file_path):
    global rawdata_dataframe
    use_list = np.arange(cfg.pcr_prediction_config['training_cycle_number'])
    if not os.path.exists(file_path):
        print(" this file path {} is not exists ".format(file_path))
        return None
    dir_path, full_file_name = os.path.split(file_path)
    file_name, extension = os.path.splitext(full_file_name)
    if extension not in ('.csv', '.xlsx'):
        print(" this file {} is not the correct file type ".format(full_file_name))
        return None
    else:
        if extension == '.csv':
            rawdata_dataframe = pd.read_csv(file_path, usecols=use_list, index_col=0, header=None)
        elif extension == '.xlsx':
            rawdata_dataframe = pd.read_excel(file_path, usecols=use_list,index_col=0, header=None)
    rawdata_array = np.array(rawdata_dataframe.iloc[1:, :]).flatten()
    return rawdata_array


def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


def pcr_prediction(raw_data):
    """
        :arg    raw_data               The PCR raw data for prediction is required, the data type is numpy list
        :return prediction_result      PCR prediction results, including predicted cycle data, etc.
    """
    y = np.array(raw_data).flatten()
    abc = []
    interpolation = cfg.pcr_prediction_config['interpolation_number']
    T = cfg.pcr_prediction_config['prediction_timestep']
    traintestsplit_ratio = cfg.pcr_prediction_config['train_test_split_ratio']
    amplification_full_cycle_number = cfg.pcr_prediction_config['amplification_full_cycle_number']
    interpolation_method = cfg.pcr_prediction_config['interpolation_method']
    training_cycle_number = len(raw_data)
    prediction_cycle_number = amplification_full_cycle_number - training_cycle_number

    N = prediction_cycle_number * interpolation
    totalinterpolation = training_cycle_number * interpolation
    x = np.linspace(1, y.shape[0], y.shape[0])
    xnew = np.linspace(1, y.shape[0], totalinterpolation)
    f = interpolate.interp1d(x, y, kind=interpolation_method)
    ynew = f(xnew)
    ynew_interpolate_dataframe = pd.DataFrame(ynew)

    scaler = MinMaxScaler(feature_range=(0, 1))
    ynew_scaled_array = scaler.fit_transform(np.array(ynew_interpolate_dataframe).reshape(-1, 1))
    training_size = int(len(ynew_scaled_array) * traintestsplit_ratio)
    test_size = len(ynew_scaled_array) - training_size
    train_data, test_data = ynew_scaled_array[0:training_size, :], ynew_scaled_array[
                                                                   training_size:len(ynew_scaled_array), :]

    time_step = T
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    ####################################################################################################################
    """
    Deep Learning Model Construction
    """

    model = Sequential()

    model.add(LSTM(1500, return_sequences=True, input_shape=(T, 1)))
    model.add(Bidirectional(LSTM(1500, return_sequences=True), merge_mode='concat'))
    model.add(Attention(step_dim=T))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    ################################################################################################################

    history = LossHistory()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=50, verbose=1,
              callbacks=[history])

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    test_predict1 = pd.DataFrame(test_predict)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Error computation
    training_squared_error = math.sqrt(mean_squared_error(y_train, train_predict))
    test_squared_error = math.sqrt(mean_squared_error(y_test, test_predict))
    look_back = T
    trainPredictPlot = numpy.empty_like(ynew_scaled_array)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(ynew_scaled_array)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(ynew_scaled_array) - 1, :] = test_predict
    x_input = test_data[len(test_data) - T:].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()

    lst_output = []
    n_steps = T
    i = 0
    while i < N:
        if len(temp_input) > T:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1

    prediction = scaler.inverse_transform(lst_output)
    print(prediction.shape)
    x = np.arange(0, prediction.shape[0], 1)
    plt.plot(x, prediction)
    prediction = pd.DataFrame(prediction)

    alldata = pd.concat([ynew_interpolate_dataframe, prediction], axis=0, ignore_index=False)  # 提取特征与标签拼接
    prediction_file_abs_path = cur_dir_path + '/save/prediction_result.csv'
    alldata.to_csv(prediction_file_abs_path, index=True)
    tranisientdata = pd.read_csv(prediction_file_abs_path)
    endpoint_value = tranisientdata.iloc[399]
    abc.append(endpoint_value)
    ynew_interpolate_array = np.array(ynew_interpolate_dataframe)
    prediction_array = np.array(prediction)
    ynew_interpolate_list = ynew_interpolate_array.flatten().tolist()
    prediction_list = prediction_array.flatten().tolist()

    return training_squared_error, test_squared_error, ynew_interpolate_list, prediction_list


if __name__ == "__main__":
    pcr_data_file_pth = cur_dir_path + '/data/raw_data.csv'

    rawdata_array = getRawdata(pcr_data_file_pth)
    print(rawdata_array)
    pcr_prediction(rawdata_array)
