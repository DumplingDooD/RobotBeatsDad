import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Defensive fetch_price for sandbox
USE_SANDBOX = True

def fetch_price():
    if USE_SANDBOX:
        return 150.0 + np.random.randn() * 2  # simulate slight price changes
    try:
        return None
    except:
        return None

# Session state initialization
if "position" not in st.session_state:
    st.session_state.position = "None"
    st.session_state.buy_price = None
    st.session_state.buy_date = None
    st.session_state.trade_log = []
    st.session_state.balance = 1000.0
    st.session_state.holdings = 0.0
    st.session_state.net_worth_log = []

price = fetch_price()
if price is None or pd.isna(price):
    st.error("❌ Failed to fetch price.")
    st.stop()

# Simulate a buy if not in position
if st.session_state.position == "None":
    st.session_state.position = "Long"
    st.session_state.buy_price = price
    st.session_state.buy_date = datetime.datetime.now()
    st.session_state.holdings = st.session_state.balance / price
    st.session_state.balance = 0
    st.session_state.trade_log.append({"date": st.session_state.buy_date, "action": "BUY", "price": price})

# Track net worth over time
net_worth = st.session_state.holdings * price if st.session_state.position == "Long" else st.session_state.balance
st.session_state.net_worth_log.append({"time": datetime.datetime.now(), "net_worth": net_worth})

st.write(f"Holding position since {st.session_state.buy_date} at ${st.session_state.buy_price:.2f}")

# Display Trade Log Table
st.subheader("🧾 Trade Log")
trade_df = pd.DataFrame(st.session_state.trade_log)
if not trade_df.empty:
    st.dataframe(trade_df)
else:
    st.info("No trades have been made yet.")

# Display Net Worth Graph
st.subheader("📈 Net Worth Over Time")
net_worth_df = pd.DataFrame(st.session_state.net_worth_log)
if not net_worth_df.empty:
    net_worth_df.set_index("time", inplace=True)
    st.line_chart(net_worth_df)
else:
    st.info("Net worth chart will appear after the first data point is recorded.")
