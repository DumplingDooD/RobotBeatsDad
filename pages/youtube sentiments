import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import pandas as pd

st.title("🎥 YouTube Sentiment Extractor for Trading Signals")

# User input
video_id = st.text_input("Enter YouTube Video ID")
target_entities = st.text_area("Entities to Track (comma separated)", "SOLANA,ETH,BITCOIN,BUY,SELL,BULLISH,BEARISH")

if st.button("🪄 Transcribe and Extract Sentiment") and video_id:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript_list])
        st.success("Transcript fetched successfully.")

        # Load sentiment pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")

        # Split transcript into chunks
        chunks = [transcript_text[i:i+512] for i in range(0, len(transcript_text), 512)]

        results = []
        for chunk in chunks:
            sentiment = sentiment_pipeline(chunk)[0]
            results.append({"text": chunk[:50] + "...", "label": sentiment['label'], "score": sentiment['score']})

        results_df = pd.DataFrame(results)
        st.subheader("📄 Sentiment on Transcript Chunks")
        st.dataframe(results_df)

        # Entity extraction
        from transformers import pipeline as ner_pipeline
        ner = ner_pipeline("ner", grouped_entities=True)
        entities = ner(transcript_text)

        tracked_entities = [e.strip().upper() for e in target_entities.split(",")]
        entity_records = []
        for ent in entities:
            entity_text = ent['word'].upper()
            if any(te in entity_text for te in tracked_entities):
                entity_records.append({"entity": ent['word'], "score": ent['score'], "label": ent['entity_group']})

        if entity_records:
            entity_df = pd.DataFrame(entity_records)
            st.subheader("🔍 Extracted Relevant Entities")
            st.dataframe(entity_df)
        else:
            st.info("No tracked entities found in this video.")

        st.success("✅ Processing complete. You can use these signals in your sentiment engine.")

    except Exception as e:
        st.error(f"Error fetching transcript or processing: {e}")

st.markdown("This tool will help you **transcribe YouTube videos, extract relevant entities, and generate sentiment signals** for your automated trading workflows.")
