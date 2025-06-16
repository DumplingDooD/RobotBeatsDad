import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Page config
st.set_page_config(page_title="Backtest Strategy", layout="wide")
st.title("📊 Backtest Sentiment Trading Strategy")

# --- User inputs ---
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.date_input("End Date", value=datetime(2024, 6, 1))
initial_capital = st.number_input("Starting Capital ($)", min_value=100, value=1000, step=100)
sentiment_threshold = st.slider("Sentiment Buy Threshold", -1.0, 1.0, 0.2, 0.1)

# --- Load historical price and sentiment data ---
@st.cache_data
def load_price_data():
    # Replace this with your real CSV or API call
    date_range = pd.date_range(start="2024-01-01", end="2024-06-01", freq="D")
    prices = np.cumsum(np.random.randn(len(date_range)) * 0.5) + 100  # Fake price series
    return pd.DataFrame({"date": date_range, "close": prices}).set_index("date")

@st.cache_data
def load_sentiment_data():
    date_range = pd.date_range(start="2024-01-01", end="2024-06-01", freq="D")
    sentiments = np.random.uniform(-1, 1, len(date_range))  # Fake sentiment
    return pd.DataFrame({"date": date_range, "sentiment": sentiments}).set_index("date")

price_df = load_price_data()
sentiment_df = load_sentiment_data()

# Filter by selected date range
price_df = price_df.loc[start_date:end_date]
sentiment_df = sentiment_df.loc[start_date:end_date]

# --- Backtesting Logic ---
def run_backtest(prices, sentiments, capital, threshold):
    capital_over_time = []
    position = 0
    trade_log = []

    for date in prices.index:
        price = prices.loc[date, "close"]
        sentiment = sentiments.loc[date, "sentiment"]

        if sentiment > threshold and position == 0:
            position = capital / price
            capital = 0
            trade_log.append((date, "BUY", price))

        elif sentiment < -threshold and position > 0:
            capital = position * price
            position = 0
            trade_log.append((date, "SELL", price))

        portfolio_value = capital + position * price
        capital_over_time.append({"date": date, "portfolio": portfolio_value})

    return pd.DataFrame(capital_over_time).set_index("date"), trade_log

# --- Run & Display ---
if st.button("Run Backtest"):
    perf_df, trades = run_backtest(price_df, sentiment_df, initial_capital, sentiment_threshold)

    st.subheader("📈 Capital Over Time")
    st.line_chart(perf_df["portfolio"])

    st.subheader("🧾 Trade Log")
    for date, action, price in trades:
        st.markdown(f"- **{action}** on `{date.date()}` @ ${price:.2f}")
