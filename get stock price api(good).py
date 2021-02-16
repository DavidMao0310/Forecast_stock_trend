import pandas as pd
pd.set_option('display.max_columns', None)
import yfinance as yf
df = yf.download("AAPL", start="2019-02-01", end="2021-02-01", interval="1d")
#df.rename(columns={'Adj Close':'adj_Close'})
print(df.head())
df.to_csv('dataset/AAPL.csv')
