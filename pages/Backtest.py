import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Engine Backtester", layout="wide")
st.title("📊 Sentiment Engine Backtest: Buy Green, Hold Yellow, Sell Red")

# User inputs
today = datetime.today()
def_start = today - timedelta(days=180)
start_date = st.date_input("Start Date", def_start)
end_date = st.date_input("End Date", today)
initial_capital = st.number_input("Starting Capital ($)", min_value=100, value=1000, step=100)

# Load SOL price data
@st.cache_data(show_spinner=False)
def load_price_data(start_date, end_date):
    cg = CoinGeckoAPI()
    from_ts = int(pd.Timestamp(start_date).timestamp())
    to_ts = int(pd.Timestamp(end_date).timestamp())
    data = cg.get_coin_market_chart_range_by_id('solana', 'usd', from_ts, to_ts)
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'close']].set_index('date').resample('D').ffill().dropna()
    return df

def add_indicators(df):
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    stoch = StochasticOscillator(df['high'] if 'high' in df.columns else df['close'],
                                 df['low'] if 'low' in df.columns else df['close'],
                                 df['close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

def generate_signal(row):
    bullish = 0
    bearish = 0
    if row['MACD'] > row['Signal']:
        bullish += 1
    else:
        bearish += 1
    if row['RSI'] < 30:
        bullish += 1
    elif row['RSI'] > 70:
        bearish += 1
    if row['Stoch_%K'] > row['Stoch_%D'] and row['Stoch_%K'] < 20:
        bullish += 1
    elif row['Stoch_%K'] < row['Stoch_%D'] and row['Stoch_%K'] > 80:
        bearish += 1
    if row['close'] < row['BB_Low']:
        bullish += 1
    elif row['close'] > row['BB_High']:
        bearish += 1
    if bullish >= 2:
        return 'Green'
    elif bearish >= 2:
        return 'Red'
    else:
        return 'Yellow'

def run_backtest(df, capital):
    position = 0
    portfolio_values = []
    trades = []
    for date, row in df.iterrows():
        price = row['close']
        signal = row['Signal_Color']
        if signal == 'Green' and position == 0:
            position = capital / price
            capital = 0
            trades.append((date, 'BUY', price))
        elif signal == 'Red' and position > 0:
            capital = position * price
            position = 0
            trades.append((date, 'SELL', price))
        portfolio = capital + position * price
        portfolio_values.append({'date': date, 'portfolio': portfolio})
    return pd.DataFrame(portfolio_values).set_index('date'), trades

if st.button("🚀 Run Backtest"):
    price_df = load_price_data(start_date, end_date)
    price_df['open'] = price_df['close'].shift(1)
    price_df['high'] = price_df[['open', 'close']].max(axis=1)
    price_df['low'] = price_df[['open', 'close']].min(axis=1)
    price_df = add_indicators(price_df)
    price_df['Signal_Color'] = price_df.apply(generate_signal, axis=1)

    perf_df, trade_log = run_backtest(price_df, initial_capital)

    st.subheader("📈 Portfolio Value Over Time")
    st.line_chart(perf_df['portfolio'])

    st.subheader("🧾 Trade Log")
    if trade_log:
        for date, action, price in trade_log:
            st.markdown(f"- **{action}** on `{date.date()}` at `${price:.2f}`")
    else:
        st.info("No trades were triggered in this period.")

    st.subheader("📊 Final Portfolio Value")
    st.write(f"${perf_df['portfolio'].iloc[-1]:,.2f}")
    
