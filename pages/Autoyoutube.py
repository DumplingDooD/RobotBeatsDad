import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime
import requests
import json

st.set_page_config(page_title="YouTube Sentiment Trader", layout="wide")
st.title("🤖 Automated YouTube Sentiment Trader with Live Engine Feed")

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

target_entities = st.text_area("Entities to Track (comma separated)", "SOLANA,ETH,BITCOIN,BUY,SELL,BULLISH,BEARISH")
api_key = st.text_input("Enter your YouTube Data API Key")


@st.cache_data(ttl=86400)
def fetch_and_analyze(api_key, channel_ids, target_entities):
    youtube = build('youtube', 'v3', developerKey=api_key)
    sentiment_pipeline = pipeline("sentiment-analysis")
    summary_records = []
    sentiments_summary = []
    
    for channel_id in channel_ids:
        request = youtube.search().list(part="snippet", channelId=channel_id, order="date", maxResults=1)
        response = request.execute()
        for item in response['items']:
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
            except Exception as e:
                st.warning(f"Transcript unavailable for {video_title} ({video_url}).")
    
    summary_df = pd.DataFrame(summary_records)
    sentiment_count = pd.Series(sentiments_summary).value_counts().to_dict()
    return summary_df, sentiment_count

if st.button("🚀 Run and Feed to Sentiment Engine") and api_key:
    try:
        summary_df, sentiment_count = fetch_and_analyze(api_key, PRESET_CHANNEL_IDS, target_entities)

        st.subheader("📊 Sentiment Table Across YouTubers")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("📈 Sentiment Summary")
        st.write(sentiment_count)

        # Feed to main sentiment engine automatically
        payload = {
            "date": str(datetime.now().date()),
            "sentiment_summary": sentiment_count,
            "detailed_sentiments": summary_df.to_dict(orient="records")
        }
        try:
            response = requests.post(engine_endpoint, json=payload)
            if response.status_code == 200:
                st.success("✅ Sentiment data successfully fed to your main trading sentiment engine.")
            else:
                st.warning(f"⚠️ Engine responded with status code {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Failed to connect to the sentiment engine: {e}")

        st.success("✅ Daily analysis complete, cached for 24 hours.")
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("This page **automatically fetches, transcribes, and analyzes the latest videos from 7 preset YouTubers, aggregates sentiment daily, and feeds the structured results directly into your main trading sentiment engine for live trade decisions.** Replace `PRESET_CHANNEL_IDS` with your YouTuber IDs and set your sentiment engine URL for seamless integration.")
