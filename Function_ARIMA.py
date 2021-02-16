import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import pmdarima as pm
from pmdarima.arima import ndiffs
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA

plt.style.use('ggplot')
pd.set_option('display.max_columns', None)


def odiff(x):
    kpss_diffs = ndiffs(x, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(x, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    return n_diffs


def lookorder(x, n_diffs):
    model = pm.auto_arima(x, start_p=0, start_q=0, max_p=15, max_q=15,
                          seasonal=False, d=n_diffs, trace=True,
                          error_action='ignore', suppress_warnings=True)
    return model


def arima_buildfit(x, order):
    model = ARIMA(x, order=order)
    t = model.fit()
    return t


def tsdisplay(y, figsize=(14, 8), lags=10):
    tmp_data = pd.Series(y)
    fig = plt.figure(figsize=figsize)
    tmp_data.plot(ax=fig.add_subplot(311), title="Time Series of Residuals", legend=False, c='seagreen')
    plot_acf(tmp_data, lags=lags, zero=False, ax=fig.add_subplot(323))
    plt.xticks(np.arange(1, lags + 1, 1.0))
    plot_pacf(tmp_data, lags=lags, zero=False, ax=fig.add_subplot(324))
    plt.xticks(np.arange(1, lags + 1, 1.0))
    qqplot(tmp_data, line='s', ax=fig.add_subplot(325), c='cornflowerblue')
    plt.title("QQ Plot")
    fig.add_subplot(326).hist(tmp_data, bins=40, density=True, range=[-5, 5])
    plt.title("Histogram")
    plt.tight_layout()
    plt.show()


def predict(fit, oridata):
    fig = plt.figure(figsize=(12, 8))
    data_size = oridata.shape[0]
    fit.plot_predict(1, data_size - 20, ax=fig.add_subplot(211), alpha=0.05, plot_insample=True)
    plt.legend().remove()
    fit.plot_predict(data_size - 30, data_size + 10, ax=fig.add_subplot(212))
    plt.tight_layout()
    plt.show()


def DoARIMA(data):
    """
        data=Any, (the target series)
    """
    ndiff = odiff(data)
    order = lookorder(data, ndiff).order
    modelfit = arima_buildfit(data, order)
    print(modelfit.summary())
    # residuals = modelfit.resid
    # tsdisplay(residuals)
    predict(modelfit, data)

if __name__ == '__main__':
    data = pd.read_csv('dataset/AAPL.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data['Close']
    DoARIMA(data)

