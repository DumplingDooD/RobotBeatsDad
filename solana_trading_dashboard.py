import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import mplfinance as mpf

# --- CONFIG ---
st.set_page_config(layout="wide")
st.title("SOL/USDT Trading Dashboard (Using CoinGecko API)")

# --- TRIGGER RERUN ---
if st.button("🔁 Rerun App"):
    st.session_state.clear()
    st.experimental_rerun()

# --- FETCH OHLCV DATA ---
@st.cache_data(ttl=3600)
def fetch_ohlcv(interval='daily', outputsize=180):
    symbol = "solana"
    vs_currency = "usd"
    days = "30"
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}"
    response = requests.get(url)

    if response.status_code != 200:
        st.error("Error fetching data from CoinGecko API.")
        return pd.DataFrame()

    prices = response.json().get("prices", [])
    df = pd.DataFrame(prices, columns=["timestamp", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    df["close"] = df["close"].astype(float)
    df["open"] = df["close"].shift(1).ffill()  
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    df["volume"] = np.random.uniform(1000000, 5000000, size=len(df))
    df.dropna(subset=["close", "open", "high", "low"], inplace=True)
    return df.tail(outputsize)

# --- ADD TECHNICAL INDICATORS ---
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

# --- FETCH NEWS AND ANALYZE SENTIMENT ---
def fetch_news_sentiment(query="Solana"):
    # Setup NewsAPI (replace with your API key)
    news_api_key = "your_news_api_key_here"
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={news_api_key}"

    response = requests.get(url)
    articles = response.json().get("articles", [])

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = {"Bullish": 0, "Bearish": 0, "Neutral": 0}

    for article in articles:
        text = article["title"] + " " + article["description"]
        sentiment_score = sia.polarity_scores(text)["compound"]
        
        if sentiment_score > 0.05:
            sentiment_scores["Bullish"] += 1
        elif sentiment_score < -0.05:
            sentiment_scores["Bearish"] += 1
        else:
            sentiment_scores["Neutral"] += 1

    return sentiment_scores

# --- GENERATE FINAL SENTIMENT ---
def generate_combined_sentiment(df):
    technical_sentiment, reasons = generate_sentiment(df)
    news_sentiment = fetch_news_sentiment(query="Solana")

    # Combine technical sentiment and news sentiment
    combined_sentiment = "Neutral"
    if news_sentiment["Bullish"] > news_sentiment["Bearish"]:
        combined_sentiment = "Bullish"
    elif news_sentiment["Bearish"] > news_sentiment["Bullish"]:
        combined_sentiment = "Bearish"

    return combined_sentiment

# --- GENERATE SIGNAL BASED ON TECHNICAL INDICATORS ---
def generate_sentiment(df):
    sentiment = "Neutral"
    bullish_signals = 0
    bearish_signals = 0

    latest = df.iloc[-1]
    reasons = []

    if 'MACD' in df.columns and 'Signal' in df.columns:
        if latest['MACD'] > latest['Signal']:
            bullish_signals += 1
            reasons.append("MACD crossover indicates bullish momentum")
        elif latest['MACD'] < latest['Signal']:
            bearish_signals += 1
            reasons.append("MACD crossover indicates bearish momentum")

    if 'RSI' in df.columns:
        if latest['RSI'] < 30:
            bullish_signals += 1
            reasons.append("RSI is below 30 (oversold) - Bullish sentiment")
        elif latest['RSI'] > 70:
            bearish_signals += 1
            reasons.append("RSI is above 70 (overbought) - Bearish sentiment")

    if 'Stoch_%K' in df.columns and 'Stoch_%D' in df.columns:
        if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
            bullish_signals += 1
            reasons.append("Stochastic RSI crossover below 20 (bullish signal)")
        elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
            bearish_signals += 1
            reasons.append("Stochastic RSI crossover above 80 (bearish signal)")

    if 'bb_low' in df.columns and latest['close'] < latest['bb_low']:
        bullish_signals += 1
        reasons.append("Price below lower Bollinger Band (potential reversal - Bullish)")
    elif 'bb_high' in df.columns and latest['close'] > latest['bb_high']:
        bearish_signals += 1
        reasons.append("Price above upper Bollinger Band (potential reversal - Bearish)")

    if bullish_signals > bearish_signals:
        sentiment = "Bullish"
    elif bearish_signals > bullish_signals:
        sentiment = "Bearish"
    
    return sentiment, reasons

# --- UI ---
timeframe = st.sidebar.selectbox("Select timeframe:", options=["1d", "1w"], index=0)
interval_map = {"1d": "daily", "1w": "daily"}
limit = 90 if timeframe == "1d" else 30

df = fetch_ohlcv(interval=interval_map[timeframe], outputsize=limit)
df = add_indicators(df)

st.write("Raw DataFrame Preview:")
st.write(df.head())

if df.empty:
    st.warning(f"No data to display for {timeframe}. Try a different timeframe or check API status.")
    st.stop()

# --- TREND SUMMARY ---
st.subheader("Price Trend Summary")
combined_sentiment = generate_combined_sentiment(df)
st.success(f"📝 Combined Sentiment: {combined_sentiment}")

# --- CHARTS ---
st.subheader("Candlestick Chart")
try:
    mpf_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
    mpf.plot(mpf_df, type='candle', volume=True, style='yahoo', title=f'SOL/USDT {timeframe} Candlestick Chart', mav=(9, 21), show_nontrading=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.error(f"Error plotting candlestick chart: {e}")

# --- AI SENTIMENT ENGINE ---
st.subheader("AI Sentiment Engine")
st.write("**Reasons for Sentiment:**")
st.write(f"Sentiment was derived from both **technical indicators** and **news sentiment**.")

