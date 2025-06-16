import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pycoingecko import CoinGeckoAPI

# --- Config ---
st.set_page_config(page_title="Backtest Strategy", layout="wide")
st.title("📊 Backtest Sentiment Trading Strategy")

# --- Define date constraints for CoinGecko ---
MAX_LOOKBACK_DAYS = 365
TODAY = datetime.today()
EARLIEST_DATE = TODAY - timedelta(days=MAX_LOOKBACK_DAYS)

# --- User inputs (default to last 180 days) ---
default_start = TODAY - timedelta(days=180)
default_end = TODAY

start_date = st.date_input("Start Date", value=default_start, min_value=EARLIEST_DATE, max_value=TODAY)
end_date = st.date_input("End Date", value=default_end, min_value=start_date, max_value=TODAY)

initial_capital = st.number_input("Starting Capital ($)", min_value=100, value=1000, step=100)
sentiment_threshold = st.slider("Sentiment Buy Threshold", -1.0, 1.0, 0.2, 0.1)

# --- Load SOL price data from CoinGecko ---
@st.cache_data(show_spinner=False)
def load_price_data(start_date, end_date):
    cg = CoinGeckoAPI()
    from_ts = int(pd.Timestamp(start_date).timestamp())
    to_ts = int(pd.Timestamp(end_date).timestamp())

    data = cg.get_coin_market_chart_range_by_id(
        id='solana',
        vs_currency='usd',
        from_timestamp=from_ts,
        to_timestamp=to_ts
    )
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['date', 'close']].set_index('date')
    df = df.resample("D").ffill()
    return df

# --- Dummy sentiment data ---
@st.cache_data(show_spinner=False)
def load_sentiment_data(dates):
    np.random.seed(42)
    scores = np.random.uniform(-1, 1, len(dates))
    return pd.DataFrame({"date": dates, "sentiment": scores}).set_index("date")

# --- Backtest logic ---
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

# --- Validation ---
if (end_date - start_date).days > MAX_LOOKBACK_DAYS:
    st.error("❌ Date range exceeds 365-day CoinGecko free API limit. Please select a shorter range.")
    st.stop()

# --- Load & Run ---
try:
    price_df = load_price_data(start_date, end_date)
    sentiment_df = load_sentiment_data(price_df.index)
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# --- Run backtest ---
if st.button("🚀 Run Backtest"):
    perf_df, trades = run_backtest(price_df, sentiment_df, initial_capital, sentiment_threshold)

    st.subheader("📈 Capital Over Time")
    st.line_chart(perf_df["portfolio"])

    st.subheader("🧾 Trade Log")
    if trades:
        for date, action, price in trades:
            st.markdown(f"- **{action}** on `{date.date()}` @ ${price:.2f}")
    else:
        st.info("No trades triggered during this time.")
