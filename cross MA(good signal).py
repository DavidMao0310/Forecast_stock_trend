import numpy as np
import pandas as pd
import talib as ta
from matplotlib import pyplot as plt

data = pd.read_csv('dataset/TSLA (30:01:21).csv')
pd.set_option('display.max_columns', None)
data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
data.set_index('Date', inplace=True)

short_window = 40
long_window = 100
signals = pd.DataFrame(index=data.index)
signals['signal'] = 0.0
signals['short_mavg'] = ta.EMA(data['Close'].values, timeperiod=short_window)
signals['long_mavg'] = ta.EMA(data['Close'].values, timeperiod=long_window)
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)
# Generate trading orders
signals['positions'] = signals['signal'].diff()
print(signals)
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Price in $')
data['Close'].plot(ax=ax1, lw=2.)
signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')
plt.show()

