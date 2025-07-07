import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import pandas as pd
from googleapiclient.discovery import build
import os

st.set_page_config(page_title="YouTube Sentiment Aggregator", layout="wide")
st.title("🤖 Automated YouTube Sentiment Engine for Trading")

# User inputs
api_key = st.text_input("Enter your YouTube Data API Key")
youtuber_channels = st.text_area("Enter YouTuber Channel IDs (one per line)")
target_entities = st.text_area("Entities to Track (comma separated)", "SOLANA,ETH,BITCOIN,BUY,SELL,BULLISH,BEARISH")

if st.button("🚀 Fetch Latest Videos and Analyze") and api_key and youtuber_channels:
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_ids = [cid.strip() for cid in youtuber_channels.split('\n') if cid.strip() != '']
        sentiment_pipeline = pipeline("sentiment-analysis")

        summary_records = []
        sentiments_summary = []

        for channel_id in channel_ids:
            request = youtube.search().list(
                part="snippet",
                channelId=channel_id,
                order="date",
                maxResults=1
            )
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
                    # Aggregate sentiment
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

        # Display table
        summary_df = pd.DataFrame(summary_records)
        st.subheader("📊 Sentiment Table Across YouTubers")
        st.dataframe(summary_df, use_container_width=True)

        # Sentiment count summary
        sentiment_count = pd.Series(sentiments_summary).value_counts().to_dict()
        st.subheader("📈 Sentiment Summary")
        st.write(sentiment_count)

        st.success("✅ Analysis complete. You can now feed these aggregated sentiments into your main sentiment engine to inform trades.")

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("This tool automatically fetches the **latest videos from specified YouTubers, transcribes them, analyzes sentiment, and aggregates signals** for your trading system.")
