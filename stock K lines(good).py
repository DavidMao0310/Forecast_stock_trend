import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv('dataset/TSLA (30:01:21).csv')
pd.set_option('display.max_columns', None)
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d')
df.set_index('Date', inplace=True)
fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.add_trace(go.Line(x=df.index,y=df['Close'],name='Close'))
fig.show()