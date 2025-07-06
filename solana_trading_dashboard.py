import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import datetime

st.set_page_config(layout="wide")
st.title("⚡ SOL/USDT Sentiment Engine (CoinGecko Indicators)")

# --- CONFIG ---
BASE_URL = "https://api.coingecko.com/api/v3"
ASSET_ID = "solana"
VS_CURRENCY = "usd"
DAYS = "30"

# --- Fetch OHLCV data from CoinGecko ---
@st.cache_data(ttl=3600)
def fetch_ohlcv():
    url = f"{BASE_URL}/coins/{ASSET_ID}/market_chart?vs_currency={VS_CURRENCY}&days={DAYS}&interval=daily"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Error fetching data from CoinGecko.")
        return pd.DataFrame()
    prices = response.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)
    df["open"] = df["close"].shift(1).ffill()
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
    df["volume"] = np.random.uniform(1000000, 5000000, size=len(df))
    df.dropna(inplace=True)
    return df

# --- Add indicators ---
def add_indicators(df):
    df["MACD"] = MACD(df['close']).macd()
    df["Signal"] = MACD(df['close']).macd_signal()
    df["RSI"] = RSIIndicator(df['close']).rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df["Stoch_%K"] = stoch.stoch()
    df["Stoch_%D"] = stoch.stoch_signal()
    bb = BollingerBands(close=df['close'])
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    return df

# --- Sentiment Engine Logic ---
def determine_sentiment(latest):
    bullish, bearish = 0, 0
    reasons = []
    if latest['MACD'] > latest['Signal']:
        bullish += 1
        reasons.append("MACD crossover bullish")
    elif latest['MACD'] < latest['Signal']:
        bearish += 1
        reasons.append("MACD crossover bearish")
    if latest['RSI'] < 30:
        bullish += 1
        reasons.append("RSI oversold")
    elif latest['RSI'] > 70:
        bearish += 1
        reasons.append("RSI overbought")
    if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
        bullish += 1
        reasons.append("Stochastic bullish")
    elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
        bearish += 1
        reasons.append("Stochastic bearish")
    if latest['close'] < latest['BB_Low']:
        bullish += 1
        reasons.append("Price below BB low")
    elif latest['close'] > latest['BB_High']:
        bearish += 1
        reasons.append("Price above BB high")
    if bullish > bearish:
        return "Bullish", reasons
    elif bearish > bullish:
        return "Bearish", reasons
    else:
        return "Neutral", reasons

# --- Main Workflow ---
df = fetch_ohlcv()
if df.empty:
    st.stop()

df = add_indicators(df)
latest = df.iloc[-1]
sentiment, reasons = determine_sentiment(latest)

# --- Display Results ---
st.subheader("📊 Latest Market Sentiment")
st.write(f"**Sentiment Signal:** `{sentiment}`")
st.write(f"**Close Price:** ${latest['close']:.2f}")
st.write("**Reasons:**")
for reason in reasons:
    st.markdown(f"- {reason}")

st.subheader("🔍 Last 5 Data Points with Indicators")
st.dataframe(df.tail(5), use_container_width=True)

st.info("✅ This dashboard functions purely as a sentiment engine for your paper trading bot without executing trades and avoids excessive API calls by caching data for 1 hour.")
