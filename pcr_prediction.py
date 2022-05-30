import numpy as np
from scipy import interpolate
import pylab as pl
import pandas as pd
import os
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
# 新加
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout


# 新加attention
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
        # self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch': [], 'epoch': []}
        # self.val_acc = {'batch':[], 'epoch':[]}
        self.saveLoss = []
        self.savevalLoss = []

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        # self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        # self.saveLoss.append(logs.get('loss'))
        # self.saveLoss.append(logs.get('loss'))

    # self.savevalLoss.append(logs.get('val_loss'))
    # self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        # self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        # self.val_acc['epoch'].append(logs.get('val_acc'))
        self.saveLoss.append(logs.get('loss'))
        self.savevalLoss.append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'r', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'b', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        # plt.show()


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


if __name__ == "__main__":
    cur_work_dir = os.getcwd()
    pcr_data_file_pth = cur_work_dir + '/Data/raw_data.csv'

    start = time.perf_counter()
    print("start:", start)
    abc = []
    for j in range(212):
        interpolation = 10
        totalinterpolation = 19 * interpolation
        y = pd.read_csv(r"D:\dell桌面\2021年11月临床数据测试\临时文件夹_阳性\阳性样本_" + str(j + 1) + ".csv")
        y = np.array(y).flatten()
        print(y.shape)
        x = np.linspace(1, y.shape[0], y.shape[0])
        print(x)
        xnew = np.linspace(1,
                           y.shape[0], totalinterpolation)
        pl.plot(x, y, "ro")
        for kind in ["nearest", "zero", "slinear", "quadratic", "cubic"]:  # 插值方式

            f = interpolate.interp1d(x, y, kind=kind)
            # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
            ynew = f(xnew)
            ynew_dataframe = pd.DataFrame(ynew)
            ynew_dataframe.to_csv(str(kind) + '.csv', index=False)

        print(totalinterpolation)

        T = 5
        N = 21 * interpolation
        trainratio = 0.95

        df = pd.read_csv('cubic.csv')
        plt.plot(df)
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))
        df = scaler.fit_transform(np.array(df).reshape(-1, 1))
        df1 = pd.DataFrame(df)
        df1.to_csv('df1.csv', index=False)

        training_size = int(len(df) * trainratio)
        test_size = len(df) - training_size
        train_data, test_data = df[0:training_size, :], df[training_size:len(df), :]


        def create_dataset(dataset, time_step):
            dataX, dataY = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return numpy.array(dataX), numpy.array(dataY)


        # reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = T
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model = Sequential()

        # model.add(LSTM(1500, return_sequences=True))
        # model.add(LSTM(1500))

        # model.add(SimpleRNN(1500, return_sequences=True))
        # model.add(SimpleRNN(1500))

        # model.add(LSTM(1600, return_sequences=True))
        # model.add(LSTM(1000))

        model.add(LSTM(1500, return_sequences=True, input_shape=(T, 1)))
        model.add(Bidirectional(LSTM(1500, return_sequences=True), merge_mode='concat'))
        # model.add(LSTM(1500))

        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')

        ##model = Sequential()
        # model.add(LSTM(1500, return_sequences=True, input_shape=(T, 1)))
        # model.add(LSTM(1500, return_sequences=True))
        model.add(Attention(step_dim=T))

        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = LossHistory()
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=50, verbose=1,
                  callbacks=[history])
        Loss = pd.DataFrame(history.saveLoss)
        Loss.to_csv('saveLoss.csv', index=False)
        valLoss = pd.DataFrame(history.savevalLoss)
        valLoss.to_csv('savevalLoss.csv', index=False)
        score = model.evaluate(X_test, y_test, verbose=0)
        history.loss_plot('epoch')

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        test_predict1 = pd.DataFrame(test_predict)
        test_predict1.to_csv('test_predict.csv', index=False)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        import math
        from sklearn.metrics import mean_squared_error

        math.sqrt(mean_squared_error(y_train, train_predict))
        math.sqrt(mean_squared_error(y_test, test_predict))
        look_back = T
        trainPredictPlot = numpy.empty_like(df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(df)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df) - 1, :] = test_predict
        # plot baseline and predictions
        # plt.plot(scaler.inverse_transform(df),'blue')
        # plt.plot(trainPredictPlot,'red')
        # plt.plot(testPredictPlot,'green')
        # plt.show()
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

        output = scaler.inverse_transform(lst_output)
        print(output.shape)

        output = pd.DataFrame(output)

        inputdata = pd.read_csv('quadratic.csv')
        alldata = pd.concat([inputdata, output], axis=0, ignore_index=False)  # 提取特征与标签拼接
        alldata.to_csv('alldata.csv', index=True)
        tranisientdata = pd.read_csv('alldata.csv')
        endpoint_value = tranisientdata.iloc[399]
        abc.append(endpoint_value)

        abcdata = pd.DataFrame(abc)
        abcdata.to_csv('abcdata.csv', index=True)
        ##losstrain=pd.read_csv('saveLoss.csv')
        ##lossval=pd.read_csv('savevalLoss.csv')
        ##allloss=pd.concat([losstrain,lossval],axis=1,ignore_index=False)
        ##allloss.to_csv('allloss.csv',index=True)

        print(j)
        j += 1

    end = time.perf_counter()
    print("\nend:", end)
    duration = end - start
    print("\nThe duration is:", duration)
