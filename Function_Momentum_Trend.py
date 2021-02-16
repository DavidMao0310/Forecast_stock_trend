import numpy as np
import pandas as pd
import talib as ta
from sklearn.preprocessing import MinMaxScaler
import functions as fc
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)

def get_MA_score(n):
    '''
    compare a target column to Close Price
    :param n: name of target column
    :return: add a dummy col in dataframe
    '''
    data['Stat_' + str(n)] = np.sign(data['Close']-data[str(n)])

def rsi_score(n):
    rsi_score = []
    for i in range(len(data[str(n)])):
        if data[str(n)][i] > 80:
            rsi_score.append(-1)
        elif data[str(n)][i] < 20:
            rsi_score.append(1)
        else:
            rsi_score.append(0)
    data['Stat_' + str(n)] = np.array(rsi_score)



def get_MA_TC(n):
    #Trend
    #compare values of MAs then return the score of MAs
    listT = [] #compare with the start value
    listC = [] #compare with the Close for long period,Avg(open,close) for short period
    for i in range(len(data[str(n)])):
        if data[str(n)][i] > data[str(n)][0]:
            listT.append(1)
        elif data[str(n)][i] < data[str(n)][0]:
            listT.append(-1)
        else:
            listT.append(0)
    data[str(n) + 'T'] = np.array(listT)

    for i in range(len(data[str(n)])):
        if str(n) in ['sma55', 'ema34']:
            if data['Close'][i] > data[str(n)][i]:
                listC.append(1)
            elif data['Close'][i] < data[str(n)][0]:
                listC.append(-1)
            else:
                listC.append(0)
        else:
            if data['Ave_open_close'][i] > data[str(n)][i]:
                listC.append(1)
            elif data['Ave_open_close'][i] < data[str(n)][0]:
                listC.append(-1)
            else:
                listC.append(0)
    data[str(n) + 'C'] = np.array(listC)


def get_sig(a):
    '''
    Get diff of MAs, then apply SMA again
    :return: dataframe add a column which is SMA again
    '''
    data['diff' + str(a[0]) + str(a[1])] = data['ema' + str(a[0])] - data['sma' + str(a[1])]
    data['Sig' + str(a[0]) + str(a[1])] = ta.SMA(data['diff' + str(a[0]) + str(a[1])].values, timeperiod=5)

# Entry and Exit Momentum
def get_mom(n):
    '''
    Compares the diff and the SMA of diff again
    :param n: names of period number(e.g 21144), means diff of 21EMA and 144SMA
    :return: score of diff MA
    '''
    listMom = []
    for i in range(len(data['diff' + str(n)])):
        if data['diff' + str(n)][i] > data['Sig' + str(n)][i]:
            listMom.append(1)
        elif data['diff' + str(n)][i] < data['Sig' + str(n)][i]:
            listMom.append(-1)
        else:
            listMom.append(0)
    data['Stat' + str(n) + 'Mom'] = np.array(listMom)


def get_StatMom():
    '''
    Method.
    Sum up diff MAs, then return the score of them
    :return: score of momentum
    '''
    data['Mom'] = data['Stat21144Mom'] + data['Stat855Mom'] + data['Stat521Mom']
    MomList = []
    for i in range(len(data['Mom'])):
        if data['Mom'][i] > 1:
            MomList.append(2)
        elif 0 < i <= 1:
            MomList.append(1)
        elif -1 <= i < 0:
            MomList.append(-1)
        elif i < -1:
            MomList.append(-2)
        else:
            MomList.append(0)
    data['Stat_Mom'] = np.array(MomList)


def Prepare(data):
    data['1d_back_close'] = data['Close'].shift(1)
    data['1d_future_close'] = data['Close'].shift(-1)
    data['2d_future_close'] = data['Close'].shift(-2)
    data['3d_future_close'] = data['Close'].shift(-3)
    data['1d_f_direction'] = np.sign(data['1d_future_close'] - data['Close'])
    data['2d_f_direction'] = np.sign(data['2d_future_close'] - data['1d_future_close'])
    data['3d_f_direction'] = np.sign(data['3d_future_close'] - data['2d_future_close'])
    #data['direction'] = 0.5 * data['1d_f_direction'] + 0.3 * data['2d_f_direction'] + 0.2 * data['3d_f_direction']
    data['direction_2'] = data['1d_f_direction']+data['2d_f_direction']
    data['Ave_open_close'] = (data['Open'] + data['Close']) * 0.5
    data.dropna(inplace=True)
    data['momi'] = ta.MOM(data['Close'].values, timeperiod=10)
    macd, macdsignal, macdhist = ta.MACD(data['Close'].values,
                                         fastperiod=12,
                                         slowperiod=26,
                                         signalperiod=9)
    data['MACD'] = macd
    data['MACDsignal'] = macdsignal
    MA_names = []
    dummies = []
    rsi_names = []
    for n in [5, 8, 13, 21, 34]:
        data['ema' + str(n)] = ta.EMA(data['Close'].values, timeperiod=n)
        MA_names = MA_names + ['ema' + str(n)]

    for n in [55, 89, 144, 233]:
        data['sma' + str(n)] = ta.SMA(data['Close'].values, timeperiod=n)
        MA_names = MA_names + ['sma' + str(n)]

    for n in [6,14,30]:
        data['rsi'+str(n)] = ta.RSI(data['Close'].values, timeperiod=n)
        rsi_names = rsi_names + ['rsi' + str(n)]


    data.dropna(inplace=True)
    for n in MA_names:
        get_MA_score(n)
        dummies = dummies + ['Stat_' + str(n)]

    for n in rsi_names:
        rsi_score(n)
    data['rsi'] = (data['Stat_rsi6']+data['Stat_rsi14']+data['Stat_rsi30'])/3

    # Determining the Trend using MAs
    data['Stst_Total'] = sum(data[str(i)] for i in dummies)
    get_MA_score('1d_back_close')
    sig_period_list = [[21, 144], [8, 55]]
    sig_list = []
    for a in sig_period_list:
        get_sig(a)
        sig_list = sig_list + ['Sig' + str(a[0]) + str(a[1])]
    data['diff521'] = data['ema5'] - data['ema21']
    data['Sig521'] = ta.SMA(data['diff521'].values, timeperiod=5)
    sig_list = sig_list + ['Sig521']
    TCs = []
    for n in ['ema5', 'ema8', 'ema13', 'ema21', 'ema34', 'sma55']:
        get_MA_TC(n)
        TCs = TCs + [str(n) + 'T'] + [str(n) + 'C']

    def get_MATrend():
        # Sum all score of MAs, then sum up and rescore, get the Moving Avg trend
        data['MATrend'] = sum(data[str(i)] for i in TCs)
        MATList = []
        for i in range(len(data['MATrend'])):
            if data['MATrend'][i] > 6:
                MATList.append(2)
            elif 0 < i <= 6:
                MATList.append(1)
            elif -6 <= i < 0:
                MATList.append(-1)
            elif i < -6:
                MATList.append(-2)
            else:
                MATList.append(0)
        data['StatMATrend'] = np.array(MATList)

    get_MATrend()
    data.dropna(inplace=True)
    for n in [21144, 855, 521]:
        get_mom(n)
    get_StatMom()
    return data


def Do_MT(data):
    data = Prepare(data)
    features_names = ['StatMATrend', 'Stat_Mom', 'rsi', 'MACD', 'Volume']
    features = data[features_names]
    targets = data['direction_2']  # our targets is 2 consecutive days of growth in Close price
    features.plot()
    targets.plot()
    plt.legend()
    plt.show()
    train_size = int(0.8 * targets.shape[0])
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_features = features[:train_size]
    train_features = scaler.fit_transform(train_features)
    train_targets = targets[:train_size]
    test_features = features[train_size:]
    test_features = scaler.fit_transform(test_features)
    test_targets = targets[train_size:]
    print(data)
    print(features)
    print(targets)
    fc.LoR(train_features, train_targets, test_features, test_targets)
    fc.SVC(train_features, train_targets, test_features, test_targets)
    fc.XGBC(train_features, train_targets, test_features, test_targets)
    fc.DTC(train_features, train_targets, test_features, test_targets)
    fc.RFC(train_features, train_targets, test_features, test_targets)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    data = pd.read_csv('dataset/AAPL.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data.set_index('Date', inplace=True)
    Do_MT(data)






