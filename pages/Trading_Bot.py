import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

st.set_page_config(layout="wide")
st.title("🤖 AI Paper Trading Bot – SOL/USDT")

# --- Fetch Price Data ---
@st.cache_data(ttl=600)
def fetch_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['solana']['usd']
    else:
        st.error("Failed to fetch price.")
        return None

# --- Fetch Sentiment (Placeholder: uses VADER on recent headlines) ---
def fetch_sentiment():
    # In real use: call your actual sentiment engine
    sia = SentimentIntensityAnalyzer()
    sample_texts = [
        "Solana hits new all-time high after ETF speculation",
        "Solana ecosystem sees growing congestion",
        "Analysts expect Solana to outperform ETH short-term"
    ]
    scores = [sia.polarity_scores(text)['compound'] for text in sample_texts]
    avg = np.mean(scores)
    if avg > 0.05:
        return "Bullish"
    elif avg < -0.05:
        return "Bearish"
    else:
        return "Neutral"

# --- Initialize Session State ---
if "position" not in st.session_state:
    st.session_state.position = "None"
    st.session_state.entry_price = None
    st.session_state.trade_log = []

# --- Trading Logic ---
def execute_trade(price, sentiment):
    log = st.session_state.trade_log
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sentiment == "Bullish" and st.session_state.position == "None":
        st.session_state.position = "Long"
        st.session_state.entry_price = price
        log.append({"time": now, "action": "BUY", "price": price})

    elif sentiment == "Bearish" and st.session_state.position == "Long":
        pnl = price - st.session_state.entry_price
        pct = (pnl / st.session_state.entry_price) * 100
        log.append({"time": now, "action": "SELL", "price": price, "pnl": pnl, "pct": pct})
        st.session_state.position = "None"
        st.session_state.entry_price = None

# --- Run Bot ---
price = fetch_price()
sentiment = fetch_sentiment()
execute_trade(price, sentiment)

# --- Display Info ---
st.subheader("📊 Current Bot Status")
st.write(f"**Current Price:** ${price:.2f}")
st.write(f"**Sentiment Signal:** `{sentiment}`")
st.write(f"**Position:** {st.session_state.position}")
if st.session_state.position == "Long":
    entry = st.session_state.entry_price
    change = price - entry
    change_pct = (change / entry) * 100
    st.success(f"Holding since ${entry:.2f} → Current PnL: ${change:.2f} ({change_pct:.2f}%)")

# --- Trade Log ---
st.subheader("🧾 Trade History")
if st.session_state.trade_log:
    df = pd.DataFrame(st.session_state.trade_log)
    st.dataframe(df[::-1], use_container_width=True)
else:
    st.info("No trades yet.")
