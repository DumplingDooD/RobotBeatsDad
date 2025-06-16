import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pycoingecko import CoinGeckoAPI

# --- Set page configuration ---
st.set_page_config(page_title="Backtest Strategy", layout="wide")
st.title("📊 Backtest Sentiment Trading Strategy")

# --- User inputs ---
start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
end_date = st.date_input("End Date", value=datetime(2024, 6, 1))
initial_capital = st.number_input("Starting Capital ($)", min_value=100, value=1000, step=100)
sentiment_threshold = st.slider("Sentiment Buy Threshold", -1.0, 1.0, 0.2, 0.1)

# --- Load SOL/USDT price data from CoinGecko ---
@st.cache_data(show_spinner=False)
def load_price_data(start_date, end_date):
    cg = CoinGeckoAPI()

    # Convert to UNIX timestamps (seconds)
    from_timestamp = int(pd.Timestamp(start_date).timestamp())
    to_timestamp = int(pd.Timestamp(end_date).timestamp())

    # Fetch market chart
    data = cg.get_coin_market_chart_range_by_id(
        id='solana',
        vs_currency='usd',
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp
    )

    prices = data['prices']  # list of [timestamp(ms), price]
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'close']].set_index('date')
    df = df.resample("D").ffill()  # Daily frequency

    return df

# --- Load dummy sentiment data ---
@st.cache_data(show_spinner=False)
def load_sentiment_data(dates):
    np.random.seed(42)  # for reproducibility
    sentiment_scores = np.random.uniform(-1, 1, len(dates))
    return pd.DataFrame({"date": dates, "sentiment": sentiment_scores}).set_index("date")

# --- Run backtest ---
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

# --- Load and display data ---
try:
    with st.spinner("Loading price data..."):
        price_df = load_price_data(start_date, end_date)

    with st.spinner("Generating sentiment data..."):
        sentiment_df = load_sentiment_data(price_df.index)

    st.success("Data loaded successfully.")

except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# --- Run backtest and plot ---
if st.button("🚀 Run Backtest"):
    perf_df, trades = run_backtest(price_df, sentiment_df, initial_capital, sentiment_threshold)

    st.subheader("📈 Capital Over Time")
    st.line_chart(perf_df["portfolio"])

    st.subheader("🧾 Trade Log")
    if trades:
        for date, action, price in trades:
            st.markdown(f"- **{action}** on `{date.date()}` @ ${price:.2f}")
    else:
        st.info("No trades were triggered in this range.")

