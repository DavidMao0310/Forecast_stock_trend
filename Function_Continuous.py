import pandas as pd
import numpy as np
import talib as ta
import functions as fc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn import linear_model as lm
pd.set_option('display.max_columns', None)

def Do_Con(data):
    feature_names = []
    for n in [14, 30, 100]:
        data['sma' + str(n)] = ta.SMA(data['Close'].values, timeperiod=n)
        data['rsi' + str(n)] = ta.RSI(data['Close'].values, timeperiod=n)
        data['ema' + str(n)] = ta.EMA(data['Close'].values, timeperiod=n)
        feature_names = feature_names + ['sma' + str(n), 'rsi' + str(n), 'ema' + str(n)]

    data['ADX'] = ta.ADX(data['High'].values,
                         data['Low'].values,
                         data['Close'].values,
                         timeperiod=14)
    macd, macdsignal, macdhist = ta.MACD(data['Close'].values,
                                         fastperiod=12,
                                         slowperiod=26,
                                         signalperiod=9)
    # Stochastic Oscillator Slow
    slowk, slowd = ta.STOCH(data['High'].values,
                            data['Low'].values,
                            data['Close'].values,
                            fastk_period=5,
                            slowk_period=3,
                            slowk_matype=0,
                            slowd_period=3,
                            slowd_matype=0)
    # Momentum Index
    mom = ta.MOM(data['Close'].values, timeperiod=10)
    # Chande Momentum Oscillator
    cmo = ta.CMO(data['Close'], timeperiod=14)
    data['MACD'] = macd
    data['MACDsignal'] = macdsignal
    data['Slowk'] = slowk
    data['Slowd'] = slowd
    data['MOM'] = mom
    data['CMO'] = cmo
    feature_names = feature_names + ['ADX', 'MACD', 'MACDsignal', 'Slowk', 'Slowd', 'MOM', 'CMO','Volume']
    data = data.dropna()
    print(data)

    features = data[feature_names]
    targets = data['1d_future_close']
    train_size = int(0.8 * targets.shape[0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = features[:train_size]
    train_features = scaler.fit_transform(train_features)
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_features = scaler.fit_transform(test_features)
    test_targets = targets[train_size:]
    fc.LM(train_features, train_targets, test_features, test_targets)
    fc.DTR(train_features, train_targets, test_features, test_targets)
    fc.RFR(train_features, train_targets, test_features, test_targets)
    fc.GBR(train_features, train_targets, test_features, test_targets)

if __name__ == '__main__':
    data = pd.read_csv('dataset/AAPL.csv')
    data['1d_future_close'] = data['Close'].shift(-1)
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data.set_index('Date', inplace=True)
    Do_Con(data)


