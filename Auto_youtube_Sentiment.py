import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime

st.set_page_config(page_title="YouTube Sentiment Trader", layout="wide")
st.title("🤖 Automated YouTube Sentiment Trader with Visual Engine Feed")

# --- Fixed 8 YouTuber Channel IDs with display names ---
YOUTUBERS = {
    "UClgJyzwGs-GyaNxUHcLZrkg": "InvestAnswers",
    "UCqK_GSMbpiV8spgD3ZGloSw": "Coin Bureau",
    "UC9ZM3N0ybRtp44-WLqsW3iQ": "Mark Moss",
    "UCFU-BE5HRJoudqIz1VDKlhQ": "CTO Larsson",
    "UCRvqjQPSeaWn-uEx-w0XOIg": "Benjamin Cowen",
    "UCtOV5M-T3GcsJAq8QKaf0lg": "Bitcoin Magazine",
    "UCpvyOqtEc86X8w8_Se0t4-w": "George Gammon",
    "UCK-zlnUfoDHzUwXcbddtnkg": "ArkInvest"
}

# --- User YouTube API Key input ---
api_key = st.text_input("Enter your YouTube Data API Key", type="password")

@st.cache_data(ttl=86400)
def fetch_and_analyze(api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    sentiment_pipeline = pipeline("sentiment-analysis")
    results = []

    for channel_id, name in YOUTUBERS.items():
        request = youtube.search().list(part="snippet", channelId=channel_id, order="date", maxResults=1)
        response = request.execute()
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            publish_time = item['snippet']['publishedAt'][:10]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = " ".join([t['text'] for t in transcript_list])
                chunks = [transcript_text[i:i+512] for i in range(0, len(transcript_text), 512)]
                sentiments = []
                for chunk in chunks:
                    result = sentiment_pipeline(chunk)[0]
                    sentiments.append(result['label'])
                sentiment_result = max(set(sentiments), key=sentiments.count)
                sentiment_icon = "🟢 Bullish" if sentiment_result == "POSITIVE" else "🔴 Bearish" if sentiment_result == "NEGATIVE" else "🟡 Neutral"
                summary_lines = transcript_text.replace('\n', ' ').split('.')[:5]
                summary = '. '.join(summary_lines).strip() + '.'
                results.append({
                    "Name": name,
                    "Video Title": video_title,
                    "Published": publish_time,
                    "URL": video_url,
                    "Summary": summary,
                    "Sentiment": sentiment_icon
                })
            except Exception:
                results.append({
                    "Name": name,
                    "Video Title": video_title,
                    "Published": publish_time,
                    "URL": video_url,
                    "Summary": "Transcript unavailable.",
                    "Sentiment": "⚪️ Unknown"
                })
    return pd.DataFrame(results)

if api_key:
    with st.spinner("🚀 Fetching latest YouTuber sentiments, transcribing, and analyzing..."):
        try:
            df = fetch_and_analyze(api_key)
            if df.empty:
                st.warning("No videos found or unable to fetch data. Please check your API key or quotas.")
            else:
                st.subheader("🎥 Latest YouTuber Sentiment Dashboard")
                for _, row in df.iterrows():
                    with st.container():
                        st.markdown(f"### [{row['Name']}]({row['URL']})")
                        st.markdown(f"**Video:** {row['Video Title']}  |  **Published:** {row['Published']}")
                        st.markdown(f"**Sentiment:** {row['Sentiment']}")
                        st.markdown(f"**Summary:** {row['Summary']}")
                        st.markdown("---")

                st.success("✅ Analysis complete and cached for 24 hours.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("🔑 Please enter your YouTube Data API key above to begin analysis.")

st.markdown("This page **automatically fetches, transcribes, analyzes, and visually displays the latest videos from your preset YouTubers with traffic light sentiment and super brief summaries for informed daily decisions.**")
