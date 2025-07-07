import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import datetime

st.title("🚦 SOL/USDT Sentiment Engine Paper Trader with Live Strategy")

# Session state
if "position" not in st.session_state:
    st.session_state.position = "None"
    st.session_state.buy_price = None
    st.session_state.buy_date = None
    st.session_state.trade_log = []
    st.session_state.balance = 1000.0
    st.session_state.holdings = 0.0
    st.session_state.net_worth_log = []

@st.cache_data(ttl=3600)
def fetch_ohlcv():
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=180&interval=daily"
    response = requests.get(url)
    prices = response.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)
    df["open"] = df["close"].shift(1)
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = np.random.uniform(1000000, 5000000, size=len(df))
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

# Load data and calculate indicators
df = fetch_ohlcv()
df = add_indicators(df)
latest = df.iloc[-1]

sentiments = {}
sentiments['MACD'] = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
sentiments['RSI'] = "Bullish" if latest['RSI'] < 30 else ("Bearish" if latest['RSI'] > 70 else "Neutral")
sentiments['Stochastic'] = "Bullish" if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20 else ("Bearish" if latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80 else "Neutral")
sentiments['Bollinger'] = "Bullish" if latest['close'] < latest['BB_Low'] else ("Bearish" if latest['close'] > latest['BB_High'] else "Neutral")

bullish_count = list(sentiments.values()).count("Bullish")
bearish_count = list(sentiments.values()).count("Bearish")

if bullish_count >= 2 and st.session_state.position == "None":
    st.session_state.position = "Long"
    st.session_state.buy_price = latest['close']
    st.session_state.buy_date = datetime.datetime.now()
    st.session_state.holdings = st.session_state.balance / latest['close']
    st.session_state.balance = 0
    st.session_state.trade_log.append({"date": st.session_state.buy_date, "action": "BUY", "price": latest['close'], "sentiments": sentiments})
    st.success(f"✅ Bought SOL at ${latest['close']:.2f} on {st.session_state.buy_date}")
elif bearish_count >= 2 and st.session_state.position == "Long":
    sell_date = datetime.datetime.now()
    st.session_state.balance = st.session_state.holdings * latest['close']
    st.session_state.holdings = 0
    st.session_state.trade_log.append({"date": sell_date, "action": "SELL", "price": latest['close'], "sentiments": sentiments})
    st.success(f"✅ Sold SOL at ${latest['close']:.2f} on {sell_date}")
    st.session_state.position = "None"
    st.session_state.buy_price = None
    st.session_state.buy_date = None

net_worth = st.session_state.holdings * latest['close'] if st.session_state.position == "Long" else st.session_state.balance
st.session_state.net_worth_log.append({"time": datetime.datetime.now(), "net_worth": net_worth})

st.write(f"Position: {st.session_state.position}")
st.write(f"Net Worth: ${net_worth:.2f}")

st.subheader("📈 Trade Log")
trade_df = pd.DataFrame(st.session_state.trade_log)
if not trade_df.empty:
    st.dataframe(trade_df)
else:
    st.info("No trades yet.")

st.subheader("📊 Net Worth Over Time")
net_worth_df = pd.DataFrame(st.session_state.net_worth_log)
if not net_worth_df.empty:
    net_worth_df.set_index("time", inplace=True)
    st.line_chart(net_worth_df)
else:
    st.info("No net worth data yet.")

trade_df.to_csv("sol_paper_trades.csv", index=False)
net_worth_df.to_csv("sol_paper_net_worth.csv")
st.success("✅ Logs saved for review.")
