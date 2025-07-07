import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import datetime

st.set_page_config(layout="wide")
st.title("🚦 SOL/USDT Advanced Sentiment Engine with Live Graphs, Traffic Light & Advice")

# --- Fetch OHLCV data ---
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

# --- Add Indicators ---
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

# --- Fetch Fear & Greed Index ---
def fetch_fear_greed():
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=1")
        value = int(response.json()['data'][0]['value'])
        if value >= 75:
            return "Extreme Greed", "Bearish"
        elif value >= 50:
            return "Greed", "Bearish"
        elif value > 25:
            return "Fear", "Bullish"
        else:
            return "Extreme Fear", "Bullish"
    except:
        return "Unavailable", "Neutral"

# --- Traffic Light Helper ---
def traffic_light(status):
    colors = {"Bullish": "🟢 Bullish", "Bearish": "🔴 Bearish", "Neutral": "🟡 Neutral"}
    return colors.get(status, "🟡 Neutral")

# --- Advice based on combined signals ---
def generate_advice(sentiments, fear_greed_sentiment):
    counts = pd.Series(sentiments.values()).value_counts()
    decision = counts.idxmax() if not counts.empty else "Neutral"
    if fear_greed_sentiment == "Bullish" and decision == "Bullish":
        return "🟢 **Advice: Consider Buying - Multiple bullish signals aligned.**"
    elif fear_greed_sentiment == "Bearish" and decision == "Bearish":
        return "🔴 **Advice: Consider Selling - Multiple bearish signals aligned.**"
    else:
        return "🟡 **Advice: Hold/Wait - Signals are mixed or neutral.**"

# --- Main Workflow ---
df = fetch_ohlcv()
df = add_indicators(df)
fear_greed_value, fear_greed_sentiment = fetch_fear_greed()

st.subheader(f"😨 Fear & Greed Index: {fear_greed_value} → {traffic_light(fear_greed_sentiment)}")

latest = df.iloc[-1]
sentiments = {}
# Classify sentiments
sentiments['MACD'] = "Bullish" if latest['MACD'] > latest['Signal'] else ("Bearish" if latest['MACD'] < latest['Signal'] else "Neutral")
sentiments['RSI'] = "Bullish" if latest['RSI'] < 30 else ("Bearish" if latest['RSI'] > 70 else "Neutral")
sentiments['Stochastic'] = "Bullish" if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20 else ("Bearish" if latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80 else "Neutral")
sentiments['Bollinger'] = "Bullish" if latest['close'] < latest['BB_Low'] else ("Bearish" if latest['close'] > latest['BB_High'] else "Neutral")

# --- Plot Indicators with explanation and traffic light ---
st.subheader("📈 Live Indicator Graphs")

fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axs[0].plot(df.index, df['close'], label='Close Price')
axs[0].plot(df.index, df['BB_High'], linestyle='--', label='BB High')
axs[0].plot(df.index, df['BB_Low'], linestyle='--', label='BB Low')
axs[0].set_title(f"Bollinger Bands {traffic_light(sentiments['Bollinger'])}")
axs[0].legend()

axs[1].plot(df.index, df['MACD'], label='MACD')
axs[1].plot(df.index, df['Signal'], label='Signal')
axs[1].set_title(f"MACD {traffic_light(sentiments['MACD'])}")
axs[1].legend()

axs[2].plot(df.index, df['RSI'], color='purple')
axs[2].axhline(70, color='red', linestyle='--')
axs[2].axhline(30, color='green', linestyle='--')
axs[2].set_title(f"RSI {traffic_light(sentiments['RSI'])}")

axs[3].plot(df.index, df['Stoch_%K'], label='%K')
axs[3].plot(df.index, df['Stoch_%D'], label='%D')
axs[3].set_title(f"Stochastic Oscillator {traffic_light(sentiments['Stochastic'])}")
axs[3].legend()

plt.tight_layout()
st.pyplot(fig)

# --- Indicator Explanations ---
st.markdown("""
### 🧠 Indicator Explanations:
- **Bollinger Bands:** Price near lower band = Bullish, near upper band = Bearish.
- **MACD:** MACD above Signal = Bullish, below = Bearish.
- **RSI:** Below 30 = Bullish, Above 70 = Bearish.
- **Stochastic:** %K > %D below 20 = Bullish, %K < %D above 80 = Bearish.
""")

# --- Advice Section ---
st.subheader("📌 Strategic Advice")
st.markdown(generate_advice(sentiments, fear_greed_sentiment))

st.success("✅ The sentiment engine now shows live graphs, clear explanations, traffic light summaries, and actionable buy/sell/hold advice for your daily trading decisions.")
