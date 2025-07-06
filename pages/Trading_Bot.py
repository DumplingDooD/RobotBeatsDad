import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Defensive fetch_price for sandbox
USE_SANDBOX = True

def fetch_price():
    if USE_SANDBOX:
        return 150.0  # static mock price for testing
    try:
        # real fetch logic here
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

price = fetch_price()
if price is None or pd.isna(price):
    st.error("❌ Failed to fetch price.")
    st.stop()

if st.session_state.position == "None":
    st.session_state.position = "Long"
    st.session_state.buy_price = price
    st.session_state.buy_date = datetime.datetime.now()
    st.session_state.holdings = st.session_state.balance / price
    st.session_state.balance = 0
    st.session_state.trade_log.append({"date": st.session_state.buy_date, "action": "BUY", "price": price})

st.write(f"Holding position since {st.session_state.buy_date} at ${st.session_state.buy_price}")
