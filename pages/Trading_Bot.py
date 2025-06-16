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
import matplotlib.pyplot as plt

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
    st.session_state.capital = 1000.0
    st.session_state.balance = 1000.0
    st.session_state.holdings = 0.0
    st.session_state.net_worth_log = []

# --- User-Defined Starting Capital ---
st.sidebar.header("🔧 Bot Settings")
initial_capital = st.sidebar.number_input("Set Initial Capital ($)", min_value=100.0, value=st.session_state.capital, step=100.0)
if initial_capital != st.session_state.capital:
    st.session_state.capital = initial_capital
    st.session_state.balance = initial_capital
    st.session_state.trade_log = []
    st.session_state.position = "None"
    st.session_state.entry_price = None
    st.session_state.holdings = 0.0
    st.session_state.net_worth_log = []

# --- Trading Logic ---
def execute_trade(price, sentiment):
    log = st.session_state.trade_log
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if sentiment == "Bullish" and st.session_state.position == "None":
        st.session_state.position = "Long"
        st.session_state.entry_price = price
        st.session_state.holdings = st.session_state.balance / price
        st.session_state.balance = 0.0
        log.append({"time": now, "action": "BUY", "price": price, "holdings": st.session_state.holdings})

    elif sentiment == "Bearish" and st.session_state.position == "Long":
        proceeds = st.session_state.holdings * price
        pnl = proceeds - st.session_state.capital
        pct = (pnl / st.session_state.capital) * 100
        st.session_state.balance = proceeds
        st.session_state.position = "None"
        st.session_state.entry_price = None
        st.session_state.holdings = 0.0
        log.append({"time": now, "action": "SELL", "price": price, "pnl": pnl, "pct": pct, "balance": st.session_state.balance})

# --- Run Bot ---
price = fetch_price()
sentiment = fetch_sentiment()
execute_trade(price, sentiment)

# --- Track Net Worth ---
if st.session_state.position == "Long":
    current_value = st.session_state.holdings * price
else:
    current_value = st.session_state.balance

st.session_state.net_worth_log.append({"time": datetime.datetime.now(), "net_worth": current_value})

# --- Display Info ---
st.subheader("📊 Current Bot Status")
st.write(f"**Current Price:** ${price:.2f}")
st.write(f"**Sentiment Signal:** `{sentiment}`")
st.write(f"**Position:** {st.session_state.position}")
st.write(f"**Starting Capital:** ${st.session_state.capital:.2f}")
st.write(f"**Current Balance:** ${st.session_state.balance:.2f}")

if st.session_state.position == "Long":
    entry = st.session_state.entry_price
    current_value = st.session_state.holdings * price
    change = current_value - st.session_state.capital
    change_pct = (change / st.session_state.capital) * 100
    st.success(f"Holding since ${entry:.2f} → Current Position Value: ${current_value:.2f} | PnL: ${change:.2f} ({change_pct:.2f}%)")

# --- Trade Log ---
st.subheader("🧾 Trade History")
if st.session_state.trade_log:
    df = pd.DataFrame(st.session_state.trade_log)
    st.dataframe(df[::-1], use_container_width=True)
else:
    st.info("No trades yet.")

# --- Net Worth Chart ---
st.subheader("📈 Capital Growth Over Time")
if st.session_state.net_worth_log:
    df_net = pd.DataFrame(st.session_state.net_worth_log)
    df_net.set_index("time", inplace=True)
    st.line_chart(df_net)
else:
    st.info("Net worth chart will appear after the first data point is recorded.")
