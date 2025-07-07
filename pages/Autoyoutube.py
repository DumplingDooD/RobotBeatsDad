import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import pandas as pd
from googleapiclient.discovery import build

st.set_page_config(page_title="YouTube Sentiment Trader", layout="wide")
st.title("🤖 Automated YouTube Sentiment Trader")

# --- Configurable preset YouTubers ---
PRESET_CHANNEL_IDS = [
    "UCXXXXXXX1",
    "UCXXXXXXX2",
    "UCXXXXXXX3",
    "UCXXXXXXX4",
    "UCXXXXXXX5",
    "UCXXXXXXX6",
    "UCXXXXXXX7",
]

target_entities = st.text_area(
    "Entities to Track (comma separated)",
    "SOLANA,ETH,BITCOIN,BUY,SELL,BULLISH,BEARISH"
)

api_key = st.text_input("Enter your YouTube Data API Key", type="password")

@st.cache_data(ttl=86400)
def fetch_and_analyze(api_key, channel_ids, target_entities):
    youtube = build('youtube', 'v3', developerKey=api_key)
    sentiment_pipeline = pipeline("sentiment-analysis")
    summary_records = []
    sentiments_summary = []

    for channel_id in channel_ids:
        request = youtube.search().list(part="snippet", channelId=channel_id, order="date", maxResults=1)
        response = request.execute()
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
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
                summary = transcript_text[:300] + "..."
                summary_records.append({
                    "Channel ID": channel_id,
                    "Video Title": video_title,
                    "Sentiment": sentiment_result,
                    "Video URL": video_url,
                    "Summary": summary
                })
                sentiments_summary.append(sentiment_result)
            except Exception:
                st.warning(f"Transcript unavailable for {video_title} ({video_url}).")

    summary_df = pd.DataFrame(summary_records)
    sentiment_count = pd.Series(sentiments_summary).value_counts().to_dict()
    return summary_df, sentiment_count

if api_key:
    with st.spinner("🚀 Fetching latest YouTuber sentiments, transcribing, and analyzing..."):
        try:
            summary_df, sentiment_count = fetch_and_analyze(api_key, PRESET_CHANNEL_IDS, target_entities)

            st.subheader("📊 Sentiment Table Across YouTubers")
            if not summary_df.empty:
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No data retrieved. Check API key or channel IDs.")

            st.subheader("📈 Sentiment Summary")
            st.write(sentiment_count)

            st.success("✅ Daily analysis complete, cached for 24 hours.")
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.info("🔑 Please enter your YouTube Data API key above to begin analysis.")

st.markdown("""
This page **automatically fetches, transcribes, analyzes, and visually displays the latest videos from your preset YouTubers with traffic light sentiment and super brief summaries for informed daily decisions.**
""")
