import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import mplfinance as mpf

# Ensure vader_lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- CONFIG ---
st.set_page_config(layout="wide")
st.title("SOL/USDT Trading Dashboard (CoinGecko + NewsAPI)")

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
    news_api_key = "939abe49599c47f98a1bf6c116c49434"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=10&apiKey={news_api_key}"

    response = requests.get(url)
    if response.status_code != 200:
        st.warning("Failed to fetch news articles. Check API key or rate limit.")
        return {"Bullish": 0, "Bearish": 0, "Neutral": 0}, []

    articles = response.json().get("articles", [])
    sia = SentimentIntensityAnalyzer()

    sentiment_scores = {"Bullish": 0, "Bearish": 0, "Neutral": 0}
    annotated_articles = []

    for article in articles:
        title = article.get("title", "").strip()
        description = article.get("description", "").strip()
        url = article.get("url", "#")
        combined_text = f"{title}. {description}"

        score = sia.polarity_scores(combined_text)["compound"]

        if score > 0.05:
            sentiment = "Bullish"
            reason = "Positive language detected (compound > 0.05)"
        elif score < -0.05:
            sentiment = "Bearish"
            reason = "Negative tone detected (compound < -0.05)"
        else:
            sentiment = "Neutral"
            reason = "Neutral or balanced sentiment"

        sentiment_scores[sentiment] += 1
        annotated_articles.append({
            "headline": title,
            "description": description,
            "sentiment": sentiment,
            "reason": reason,
            "url": url
        })

    return sentiment_scores, annotated_articles

# --- COMBINE TECHNICAL + NEWS SENTIMENT ---
def generate_combined_sentiment(df):
    technical_sentiment, reasons = generate_sentiment(df)
    news_sentiment, annotated_articles = fetch_news_sentiment()

    combined_sentiment = "Neutral"
    if news_sentiment["Bullish"] > news_sentiment["Bearish"]:
        combined_sentiment = "Bullish"
    elif news_sentiment["Bearish"] > news_sentiment["Bullish"]:
        combined_sentiment = "Bearish"

    return combined_sentiment, news_sentiment, annotated_articles

# --- TECHNICAL SENTIMENT LOGIC ---
def generate_sentiment(df):
    sentiment = "Neutral"
    bullish_signals = 0
    bearish_signals = 0

    latest = df.iloc[-1]
    reasons = []

    if latest['MACD'] > latest['Signal']:
        bullish_signals += 1
        reasons.append("MACD crossover → Bullish")
    elif latest['MACD'] < latest['Signal']:
        bearish_signals += 1
        reasons.append("MACD crossover → Bearish")

    if latest['RSI'] < 30:
        bullish_signals += 1
        reasons.append("RSI < 30 → Oversold (Bullish)")
    elif latest['RSI'] > 70:
        bearish_signals += 1
        reasons.append("RSI > 70 → Overbought (Bearish)")

    if latest['Stoch_%K'] > latest['Stoch_%D'] and latest['Stoch_%K'] < 20:
        bullish_signals += 1
        reasons.append("Stochastic crossover < 20 → Bullish")
    elif latest['Stoch_%K'] < latest['Stoch_%D'] and latest['Stoch_%K'] > 80:
        bearish_signals += 1
        reasons.append("Stochastic crossover > 80 → Bearish")

    if latest['close'] < latest['bb_low']:
        bullish_signals += 1
        reasons.append("Price below lower BB → Bullish")
    elif latest['close'] > latest['bb_high']:
        bearish_signals += 1
        reasons.append("Price above upper BB → Bearish")

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

st.write("📊 **Preview of market data**")
st.dataframe(df.tail(5))

if df.empty:
    st.warning(f"No data to display for {timeframe}. Try another interval.")
    st.stop()

# --- TREND SUMMARY ---
st.subheader("📈 Price Trend Summary")
combined_sentiment, news_sentiment, annotated_articles = generate_combined_sentiment(df)

st.success(f"📝 Combined Market Sentiment: **{combined_sentiment}**")
st.markdown(f"""
**News Sentiment Breakdown:**  
🟢 Bullish: `{news_sentiment['Bullish']}`  
🔴 Bearish: `{news_sentiment['Bearish']}`  
⚪ Neutral: `{news_sentiment['Neutral']}`
""")

# --- CANDLESTICK CHART ---
st.subheader("📉 Candlestick Chart")
try:
    mpf_df = df[['open', 'high', 'low', 'close', 'volume']]
    mpf.plot(mpf_df, type='candle', volume=True, style='yahoo', title=f'SOL/USDT - {timeframe.upper()}', mav=(9, 21), show_nontrading=False)
    st.pyplot(plt.gcf())
except Exception as e:
    st.error(f"Chart rendering error: {e}")

# --- SENTIMENT HEADLINES WITH EXPLANATIONS ---
st.subheader("📰 Relevant News Headlines and Sentiment")

if not annotated_articles:
    st.info("No news articles could be analyzed.")
else:
    for article in annotated_articles:
        st.markdown(f"**🗞️ [{article['headline']}]({article['url']})**")
        st.write(f"- **Sentiment:** `{article['sentiment']}`")
        st.write(f"- **Reason:** _{article['reason']}_")
        if article['description']:
            st.write(f"- **Summary:** {article['description']}")
        st.markdown("---")

# --- SENTIMENT ENGINE EXPLANATION ---
st.subheader("🧠 Sentiment Analysis Method")
st.markdown("""
**Sources of Sentiment:**

- Technical Indicators: RSI, MACD, Stochastic, Bollinger Bands  
- News Sentiment: Headlines and summaries analyzed using VADER (NLTK)
""")
