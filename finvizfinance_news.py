import pandas as pd
from finvizfinance.quote import finvizfinance

pd.set_option('display.max_columns', None)
stock = finvizfinance('NKE')
news_df = stock.TickerNews()
print(news_df)

