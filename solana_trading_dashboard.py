import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

# --- CONFIG ---
st.set_page_config(layout="wide")
st.title("SOL/USD Trading Dashboard")

# --- FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_ohlcv(interval='1d', limit=180):
    url = f"https://api.binance.com/api/v3/klines?symbol=SOLUSDT&interval={interval}&limit={limit}"
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', '_', '_', '_', '_', '_', '_'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def add_indicators(df):
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    rsi = RSIIndicator(df['close'])
    df['RSI'] = rsi.rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    return df

# --- UI ---
timeframe = st.sidebar.selectbox("Select timeframe:", options=["1h", "1d", "1w"], index=1)
interval_map = {"1h": "1h", "1d": "1d", "1w": "1w"}
df = fetch_ohlcv(interval=interval_map[timeframe])
df = add_indicators(df)

# --- CHARTS ---
st.subheader("Price Chart")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df['close'], label='Close Price', color='black')
ax.set_title('SOL/USD Close Price')
ax.legend()
st.pyplot(fig)

st.subheader("MACD")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(df.index, df['MACD'], label='MACD', color='blue')
ax.plot(df.index, df['Signal'], label='Signal', color='orange')
ax.fill_between(df.index, df['MACD'] - df['Signal'], 0, color='gray', alpha=0.3)
ax.legend()
st.pyplot(fig)

st.subheader("RSI")
fig, ax = plt.subplots(figsize=(12, 2))
ax.plot(df.index, df['RSI'], label='RSI', color='green')
ax.axhline(70, color='red', linestyle='--', linewidth=0.5)
ax.axhline(30, color='blue', linestyle='--', linewidth=0.5)
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

st.subheader("Stochastic RSI")
fig, ax = plt.subplots(figsize=(12, 2))
ax.plot(df.index, df['Stoch_%K'], label='%K', color='purple')
ax.plot(df.index, df['Stoch_%D'], label='%D', color='magenta')
ax.axhline(80, color='red', linestyle='--', linewidth=0.5)
ax.axhline(20, color='blue', linestyle='--', linewidth=0.5)
ax.set_ylim(0, 100)
ax.legend()
st.pyplot(fig)

# --- GPT SIGNAL (Placeholder) ---
st.subheader("AI Signal Engine (Coming Soon)")
st.info("This section will analyze indicator data and return a Buy / Hold / Sell recommendation based on price theory.")
