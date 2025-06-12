import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import mplfinance as mpf

# --- CONFIG ---
st.set_page_config(layout="wide")
st.title("SOL/USDT Trading Dashboard (Using CoinGecko API)")

# --- TRIGGER RERUN ---
if st.button("🔁 Rerun App"):
    st.session_state.clear()
    st.experimental_rerun()

# --- FUNCTIONS ---

@st.cache_data(ttl=3600)
def fetch_ohlcv(interval='daily', outputsize=180):
    symbol = "solana"
    vs_currency = "usd"
    days = "30"
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}"
    response = requests.get(url)

    st.write("🌐 CoinGecko API URL:", url)
    st.write("🔄 Status Code:", response.status_code)

    try:
        json_data = response.json()
        st.write("🧎 Full JSON Response:", json_data)
    except Exception as e:
        st.error(f"Error decoding JSON: {e}")
        return pd.DataFrame()

    if response.status_code != 200 or "prices" not in json_data:
        st.error("Error fetching data from CoinGecko API.")
        return pd.DataFrame()

    prices = json_data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)

    # Forward fill NaN values instead of using bfill
    df["open"] = df["close"].shift(1).ffill()  
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = np.random.uniform(1000000, 5000000, size=len(df))

    # Remove rows with any NaN values in key columns
    df.dropna(subset=["close", "open", "high", "low"], inplace=True)

    return df.tail(outputsize)

def add_indicators(df):
    try:
        macd = MACD(df['close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
    except Exception as e:
        st.error(f"MACD calculation error: {e}")

    try:
        rsi = RSIIndicator(df['close'])
        df['RSI'] = rsi.rsi()
    except Exception as e:
        st.error(f"RSI calculation error: {e}")

    try:
        stoch = StochasticOscillator(df['high'], df['low'], df['close'])
        df['Stoch_%K'] = stoch.stoch()
        df['Stoch_%D'] = stoch.stoch_signal()
    except Exception as e:
        st.error(f"Stochastic calculation error: {e}")

    try:
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
    except Exception as e:
        st.error(f"Bollinger Bands error: {e}")

    return df

def generate_sentiment(df):
    sentiment = "Neutral"
    bullish_signals = 0
    bearish_signals = 0

    if df.empty or df.isnull().values.any():
        return "Neutral", []

    latest = df.iloc[-1]
    reasons = []

    # MACD Sentiment
    if 'MACD' in df.columns and 'Signal' in df.columns:
        if latest['MACD'] > latest['Signal']:
            bullish_signals += 1
            reasons.append("MACD crossover indicates bullish momentum")
        elif latest['MACD'] < latest['Signal']:
            bearish_signals += 1
            reasons.append("MACD crossover indicates bearish momentum")

    # RSI Sentiment
    if 'RSI' in df.columns:
        if latest['RSI'] < 30:
            bullish_signals += 1
            reasons.append("RSI is below 30 (oversold) - Bullish sentiment")
        elif latest['RSI'] > 70:
            bearish_signals += 1
            reasons.append("RSI is above 70 (overbought) - Bearish sentiment")

    # Stochastic RSI Sentiment
    if 'Stoch_%K' in df.columns and 'Stoch_%D' in df.columns:
        if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
            bullish_signals += 1
            reasons.append("Stochastic RSI crossover below 20 (bullish signal)")
        elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
            bearish_signals += 1
            reasons.append("Stochastic RSI crossover above 80 (bearish signal)")

    # Bollinger Bands Sentiment
    if 'bb_low' in df.columns and latest['close'] < latest['bb_low']:
        bullish_signals += 1
        reasons.append("Price below lower Bollinger Band (potential reversal - Bullish)")
    elif 'bb_high' in df.columns and latest['close'] > latest['bb_high']:
        bearish_signals += 1
        reasons.append("Price above upper Bollinger Band (potential reversal - Bearish)")

    # Sentiment Decision
    if bullish_signals > bearish_signals:
        sentiment = "Bullish"
    elif bearish_signals > bullish_signals:
        sentiment = "Bearish"
    
    return sentiment, reasons

# --- UI ---
timeframe = st.sidebar.selectbox("Select timeframe:", options=["1d", "1w"], index=0)
interval_map = {"1d": "daily", "1w": "daily"}
limit = 90 if timeframe == "1d" else 30

df = fetch_ohlcv(interval=interval_map[timeframe], outputsize=limit)

st.write("Raw DataFrame Preview:")
st.write(df.head())

if df.empty:
    st.warning(f"No data to display for {timeframe}. Try a different timeframe or check API status.")
    st.stop()

df = add_indicators(df)

if df['close'].isnull().all():
    st.warning("All close prices are NaN — check API response format.")

# --- TREND SUMMARY ---
st.subheader("Price Trend Summary")
trend_summary = summarize_trend(df['close'])
st.write(trend_summary)

# --- CHARTS ---
st.subheader("Candlestick Chart")
try:
    mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf.plot(mpf_df, type='candle', volume=True, style='yahoo', title=f'SOL/USDT {timeframe} Candlestick Chart', mav=(9, 21), show_nontrading=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.error(f"Error plotting candlestick chart: {e}")

if 'MACD' in df.columns:
    st.subheader("MACD")
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df.index, df['MACD'], label='MACD', color='blue')
    if 'Signal' in df.columns:
        ax.plot(df.index, df['Signal'], label='Signal', color='orange')
        ax.fill_between(df.index, df['MACD'] - df['Signal'], 0, color='gray', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

if 'RSI' in df.columns:
    st.subheader("RSI")
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(df.index, df['RSI'], label='RSI', color='green')
    ax.axhline(70, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(30, color='blue', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)

if 'Stoch_%K' in df.columns:
    st.subheader("Stochastic RSI")
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(df.index, df['Stoch_%K'], label='%K', color='purple')
    ax.plot(df.index, df['Stoch_%D'], label='%D', color='magenta')
    ax.axhline(80, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(20, color='blue', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    ax.legend()
    st.pyplot(fig)

# --- AI Sentiment Engine ---
st.subheader("AI Sentiment Engine")
sentiment, reasons = generate_sentiment(df)

if sentiment == "Neutral":
    st.warning("Unable to determine sentiment — data may be insufficient or indicators are mixed.")
else:
    st.success(f"📝 Sentiment: {sentiment}")
    st.write("**Reasons:**")
    for reason in reasons:
        st.markdown(f"- {reason}")

    st.markdown("### 💡 Recommendation")
    if sentiment == "Bullish":
        st.markdown("> Indicators suggest **bullish momentum** — it might be a good time to **buy** or hold positions.")
    elif sentiment == "Bearish":
        st.markdown("> Indicators suggest **bearish momentum** — consider securing gains or **selling**.")
    else:
        st.markdown("> Indicators show mixed signals. Consider a **neutral** stance or **holding**.")
