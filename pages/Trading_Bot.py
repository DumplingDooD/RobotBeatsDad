import streamlit as st
import pandas as pd
import numpy as np
import requests
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import matplotlib.pyplot as plt
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

st.set_page_config(layout="wide")
st.title("🚦 SOL/USDT Sentiment Engine Paper Trader with Auto P&L Email and Strategy Insights")

# Session state initialization
if "position" not in st.session_state:
    st.session_state.position = "None"
    st.session_state.buy_price = None
    st.session_state.buy_date = None
    st.session_state.trade_log = []
    st.session_state.balance = 1000.0
    st.session_state.holdings = 0.0
    st.session_state.net_worth_log = []

@st.cache_data(ttl=3600)
def fetch_ohlcv():
    url = "https://api.coingecko.com/api/v3/coins/solana/market_chart?vs_currency=usd&days=180&interval=daily"
    response = requests.get(url)
    prices = response.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)
    df["open"] = df["close"].shift(1)
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = np.random.uniform(1000000, 5000000, size=len(df))
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    macd = MACD(df['close'])
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(df['close']).rsi()
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['Stoch_%K'] = stoch.stoch()
    df['Stoch_%D'] = stoch.stoch_signal()
    bb = BollingerBands(df['close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

df = fetch_ohlcv()
df = add_indicators(df)
latest = df.iloc[-1]

sentiments = {}
sentiments['MACD'] = "Bullish" if latest['MACD'] > latest['Signal'] else "Bearish"
sentiments['RSI'] = "Bullish" if latest['RSI'] < 30 else ("Bearish" if latest['RSI'] > 70 else "Neutral")
sentiments['Stochastic'] = "Bullish" if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20 else ("Bearish" if latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80 else "Neutral")
sentiments['Bollinger'] = "Bullish" if latest['close'] < latest['BB_Low'] else ("Bearish" if latest['close'] > latest['BB_High'] else "Neutral")

bullish_count = list(sentiments.values()).count("Bullish")
bearish_count = list(sentiments.values()).count("Bearish")

if bullish_count >= 2:
    traffic_signal = "🟢 Green (Buy)"
elif bearish_count >= 2:
    traffic_signal = "🔴 Red (Sell)"
else:
    traffic_signal = "🟡 Yellow (Hold)"

st.subheader(f"🚦 Traffic Light Signal: {traffic_signal}")

if traffic_signal.startswith("🟢") and st.session_state.position == "None":
    st.session_state.position = "Long"
    st.session_state.buy_price = latest['close']
    st.session_state.buy_date = datetime.datetime.now()
    st.session_state.holdings = st.session_state.balance / latest['close']
    st.session_state.balance = 0
    st.session_state.trade_log.append({"date": st.session_state.buy_date, "action": "BUY", "price": latest['close']})
    st.success(f"✅ Bought SOL at ${latest['close']:.2f}")
elif traffic_signal.startswith("🔴") and st.session_state.position == "Long":
    sell_date = datetime.datetime.now()
    st.session_state.balance = st.session_state.holdings * latest['close']
    st.session_state.holdings = 0
    st.session_state.trade_log.append({"date": sell_date, "action": "SELL", "price": latest['close']})
    st.success(f"✅ Sold SOL at ${latest['close']:.2f}")
    st.session_state.position = "None"
    st.session_state.buy_price = None
    st.session_state.buy_date = None

net_worth = st.session_state.holdings * latest['close'] if st.session_state.position == "Long" else st.session_state.balance
st.session_state.net_worth_log.append({"time": datetime.datetime.now(), "net_worth": net_worth})

st.subheader("📈 Net Worth Over Time")
net_worth_df = pd.DataFrame(st.session_state.net_worth_log)
if not net_worth_df.empty:
    net_worth_df.set_index("time", inplace=True)
    st.line_chart(net_worth_df)

st.subheader("🧾 Trade Log")
trade_df = pd.DataFrame(st.session_state.trade_log)
if not trade_df.empty:
    st.dataframe(trade_df)
else:
    st.info("No trades executed yet.")

# Daily P&L Calculation
today = datetime.datetime.now().date()
today_trades = [t for t in st.session_state.trade_log if pd.to_datetime(t['date']).date() == today]
total_pnl = 0
if today_trades:
    buy_prices = [t['price'] for t in today_trades if t['action'] == 'BUY']
    sell_prices = [t['price'] for t in today_trades if t['action'] == 'SELL']
    total_pnl = sum(sell_prices) - sum(buy_prices)

st.subheader("📊 Daily P&L")
st.write(f"Today's P&L: ${total_pnl:.2f}")

# Strategy Insights
if not trade_df.empty:
    sells = trade_df[trade_df['action'] == 'SELL']['price'].reset_index(drop=True)
    buys = trade_df[trade_df['action'] == 'BUY']['price'].reset_index(drop=True)
    if len(buys) > 0 and len(sells) > 0:
        wins = sells > buys[:len(sells)]
        win_rate = wins.sum() / len(wins) * 100
        st.subheader("📌 Strategy Insights")
        st.write(f"Win Rate: {win_rate:.2f}%")

# Email Notification

def send_email_report():
    sender_email = os.environ["SMTP_EMAIL"]
    password = os.environ["SMTP_PASSWORD"]
    receiver_email = "Charlee1289@gmail.com"
    message = MIMEMultipart()
    message["Subject"] = "Daily P&L Report"
    message["From"] = sender_email
    message["To"] = receiver_email
    body = f"Today's P&L: ${total_pnl:.2f}\nNet Worth: ${net_worth:.2f}"
    message.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        st.success("📧 Email report sent successfully!")
    except Exception as e:
        st.error(f"Email failed: {e}")

# Uncomment to enable sending email automatically after environment variables are set
# send_email_report()

# Save logs
trade_df.to_csv("sol_paper_trades.csv", index=False)
net_worth_df.to_csv("sol_paper_net_worth.csv")
st.success("✅ Logs saved and ready for your daily analysis.")
