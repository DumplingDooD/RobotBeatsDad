import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
import mplfinance as mpf

# --- CONFIG ---
st.set_page_config(layout="wide")
st.title("SOL/USDT Trading Dashboard")

# --- FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_ohlcv(interval='1day', outputsize=180):
    api_key = "0e4cd87767fc47e9ac28cdc773b18bc5"  # Your Twelve Data API key
    symbol = "SOL/USDT"
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={api_key}"
    response = requests.get(url)

    # Debug info to investigate data issue
    st.write("📡 Twelve Data API URL:", url)
    st.write("🔄 Status Code:", response.status_code)

    try:
        json_data = response.json()
        st.write("🧾 Full JSON Response:", json_data)
    except Exception as e:
        st.error(f"Error decoding JSON: {e}")
        return pd.DataFrame()

    if response.status_code != 200 or "values" not in json_data:
        st.error("Error fetching data from Twelve Data API.")
        return pd.DataFrame()

    df = pd.DataFrame(json_data["values"])
    df.columns = [col.lower() for col in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    df = df.astype(float)
    return df

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

    return df

def generate_signal(df):
    signal = "Hold"
    reasons = []

    if df.empty or df.isnull().values.any():
        return "Insufficient data", []

    latest = df.iloc[-1]

    if 'MACD' in df.columns and 'Signal' in df.columns:
        if latest['MACD'] > latest['Signal']:
            reasons.append("MACD crossover indicates bullish momentum")
        elif latest['MACD'] < latest['Signal']:
            reasons.append("MACD crossover indicates bearish momentum")

    if 'RSI' in df.columns:
        if latest['RSI'] < 30:
            reasons.append("RSI is below 30 (oversold)")
        elif latest['RSI'] > 70:
            reasons.append("RSI is above 70 (overbought)")

    if 'Stoch_%K' in df.columns and 'Stoch_%D' in df.columns:
        if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
            reasons.append("Stochastic RSI crossover below 20 (bullish signal)")
        elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
            reasons.append("Stochastic RSI crossover above 80 (bearish signal)")

    buy_signals = ["bullish" in r or "oversold" in r for r in reasons]
    sell_signals = ["bearish" in r or "overbought" in r for r in reasons]

    if sum(buy_signals) >= 2:
        signal = "Buy"
    elif sum(sell_signals) >= 2:
        signal = "Sell"

    return signal, reasons

# --- UI ---
timeframe = st.sidebar.selectbox("Select timeframe:", options=["1h", "1d", "1w"], index=1)
interval_map = {"1h": "1h", "1d": "1day", "1w": "1week"}
limit = 180 if timeframe == "1h" else 90 if timeframe == "1d" else 52
df = fetch_ohlcv(interval=interval_map[timeframe], outputsize=limit)

# Debugging preview
st.write("Raw DataFrame Preview:")
st.write(df.head())

if df.empty:
    st.warning(f"No data to display for {timeframe}. Try a different timeframe or check API status.")
    st.stop()

# Add indicators only if data exists
df = add_indicators(df)

if df['close'].isnull().all():
    st.warning("All close prices are NaN — check API response format.")

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

# --- AI SIGNAL ENGINE ---
st.subheader("AI Signal Engine")
signal, reasons = generate_signal(df)

if signal == "Insufficient data":
    st.warning("Unable to generate signal due to missing or invalid data.")
else:
    st.success(f"🔔 Signal: {signal}")
    st.write("**Reasons:**")
    for reason in reasons:
        st.markdown(f"- {reason}")
