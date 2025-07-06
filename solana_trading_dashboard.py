import streamlit as st
import pandas as pd
import numpy as np
import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands

st.set_page_config(layout="wide")
st.title("🪐 Sandbox Sentiment-Driven Paper Trading Bot")

# Sandbox toggle
USE_SANDBOX = True

# Generate synthetic OHLCV data for sandbox testing
def generate_synthetic_ohlcv(rows=100):
    base_price = 150
    dates = pd.date_range(end=datetime.datetime.now(), periods=rows, freq='H')
    price = base_price + np.cumsum(np.random.randn(rows))
    df = pd.DataFrame({"close": price}, index=dates)
    df["open"] = df["close"].shift(1).fillna(method='bfill')
    df["high"] = df[["open", "close"]].max(axis=1) + np.random.rand(rows)
    df["low"] = df[["open", "close"]].min(axis=1) - np.random.rand(rows)
    df["volume"] = np.random.uniform(1000, 5000, size=rows)
    return df

# Add technical indicators
def add_indicators(df):
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    
    rsi = RSIIndicator(df['close'])
    df['RSI'] = rsi.rsi()

    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()

    bb = BollingerBands(close=df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    return df

# Trading logic using indicators
def determine_sentiment(latest):
    bullish, bearish = 0, 0
    reasons = []
    if latest['MACD'] > latest['Signal']:
        bullish += 1
        reasons.append("MACD bullish")
    elif latest['MACD'] < latest['Signal']:
        bearish += 1
        reasons.append("MACD bearish")
    if latest['RSI'] < 30:
        bullish += 1
        reasons.append("RSI oversold")
    elif latest['RSI'] > 70:
        bearish += 1
        reasons.append("RSI overbought")
    if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
        bullish += 1
        reasons.append("Stoch bullish")
    elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
        bearish += 1
        reasons.append("Stoch bearish")
    if latest['close'] < latest['bb_low']:
        bullish += 1
        reasons.append("Price below BB low")
    elif latest['close'] > latest['bb_high']:
        bearish += 1
        reasons.append("Price above BB high")
    if bullish > bearish:
        return "Bullish", reasons
    elif bearish > bullish:
        return "Bearish", reasons
    else:
        return "Neutral", reasons

# Initialize session state
if "position" not in st.session_state:
    st.session_state.position = "None"
    st.session_state.balance = 1000.0
    st.session_state.holdings = 0.0
    st.session_state.trade_log = []
    st.session_state.net_worth_log = []

# Main sandbox workflow
if USE_SANDBOX:
    df = generate_synthetic_ohlcv()
    df = add_indicators(df)
    latest = df.iloc[-1]
    sentiment, reasons = determine_sentiment(latest)
    price = latest['close']

    now = datetime.datetime.now()
    if sentiment == "Bullish" and st.session_state.position == "None":
        st.session_state.position = "Long"
        st.session_state.holdings = st.session_state.balance / price
        st.session_state.balance = 0
        st.session_state.trade_log.append({"time": now, "action": "BUY", "price": price, "holdings": st.session_state.holdings})
    elif sentiment == "Bearish" and st.session_state.position == "Long":
        st.session_state.balance = st.session_state.holdings * price
        st.session_state.holdings = 0
        st.session_state.position = "None"
        st.session_state.trade_log.append({"time": now, "action": "SELL", "price": price, "balance": st.session_state.balance})
    net_worth = st.session_state.balance + st.session_state.holdings * price
    st.session_state.net_worth_log.append({"time": now, "net_worth": net_worth})

    # Display outputs
    st.write(f"**Current Price:** ${price:.2f}")
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Reasons:** {', '.join(reasons)}")
    st.write(f"**Position:** {st.session_state.position}")
    st.write(f"**Net Worth:** ${net_worth:.2f}")

    if st.session_state.trade_log:
        st.subheader("🧾 Trade Log")
        st.dataframe(pd.DataFrame(st.session_state.trade_log))

    if st.session_state.net_worth_log:
        st.subheader("📈 Net Worth Over Time")
        df_net = pd.DataFrame(st.session_state.net_worth_log).set_index("time")
        st.line_chart(df_net)

    st.success("✅ Sandbox trading simulation using your real indicators is active without using API credits.")
