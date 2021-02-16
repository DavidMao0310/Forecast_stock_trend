import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly_express as px
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot')


def creat_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i: (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def make_doo(dataset1, look_back):
    do_set = []
    make_do = dataset1[len(dataset1) - 30 - look_back:]
    for i in range(look_back, look_back + 30 + 1):
        do_set.append(make_do[i - look_back:i])
    do_set = np.array(do_set)
    return do_set


def DoLSTM(data,lags):
    """
        data=Any, (the target series)
        lags=Any int, (the previous days related to today's price)
    """
    dataset = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    train_size = int(len(dataset) * 0.8)
    train, test = dataset[0: train_size], dataset[train_size: len(dataset)]
    look_back = lags

    trainX, trainY = creat_dataset(train, look_back)
    testX = []
    inputs = dataset[len(dataset) - len(test) - look_back:]
    for i in range(look_back, look_back + len(test)):
        testX.append(inputs[i - look_back:i])
    testX = np.array(testX)
    testY = data[train_size: len(dataset)].values
    model = Sequential()
    model.add(LSTM(input_dim=1, units=50, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(input_dim=50, units=100, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(input_dim=100, units=200, return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(LSTM(300, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(100))
    model.add(Dense(units=1))

    model.add(Activation('relu'))
    start = time.time()
    model.compile(loss='MSE', optimizer='Adam')
    model.summary()

    history = model.fit(trainX, trainY, batch_size=64, epochs=50,
                        validation_split=0.1, verbose=2)
    print('compilatiom time:', time.time() - start)
    trainPredict = model.predict(trainX)
    print('len_trainX', len(trainX))
    print('len_trainY', len(trainY))
    print(len(trainPredict))
    testPredict = model.predict(testX)
    print('len_testX', len(testX))
    print('len_testY', len(testY))

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform(testY)
    trainScore = np.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
    print('Train Sccore: RMSE', trainScore)
    print('Train Sccore: R^2', r2_score(trainY, trainPredict[:, 0]))

    testScore = np.sqrt(mean_squared_error(testY, testPredict[:, 0]))
    print('Test Sccore: RMSE', testScore)
    print('Test Sccore: R^2', r2_score(testY, testPredict[:, 0]))

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

    test_plot = data[train_size: len(dataset)]
    test_plot['Predict'] = np.array(testPredict[:, 0])
    test_plot['Predict'].plot()
    test_plot['Close'].plot()
    plt.title('model test performance')
    plt.show()
    fig = px.line(test_plot, x=test_plot.index, y=[test_plot['Predict'], test_plot['Close']],
                  title='model test performance')
    fig.show()

    do_plot = data[train_size: len(dataset)]
    do_plot.reset_index(drop=False, inplace=True)
    do_set1 = make_doo(scaler.fit_transform(pd.DataFrame(do_plot['Close']).values), look_back)
    print(scaler.inverse_transform(model.predict(do_set1))[-1][0])

    for i in range(3):
        dataset1 = scaler.fit_transform(pd.DataFrame(do_plot['Close']).values)
        do_set = make_doo(dataset1, look_back)
        do_plot.loc[len(dataset1) + i] = [do_plot.iloc[-1, 0] + timedelta(days=1),
                                          scaler.inverse_transform(model.predict(do_set))[-1][0]]

    print(do_plot)
    do_plot.set_index('Date', inplace=True)
    do_plot.plot()
    plt.scatter(do_plot[-3:].index, do_plot[-3:].values, color='blue')
    plt.title('LSTM forecast')
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv('dataset/AAPL.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = pd.DataFrame(data['Close'])
    print(data)
    DoLSTM(data,3)
