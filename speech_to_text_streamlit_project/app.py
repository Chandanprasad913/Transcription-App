import os
import streamlit as st
from realtime_record import record_audio
from realtime_inference import transcribe

BASE_DIR = os.path.dirname(__file__)
AUDIO_PATH = os.path.join(BASE_DIR, "live.wav")

st.set_page_config(page_title="Speech to Text Demo", page_icon="🎤")
st.title("🎤 Real-Time Speech to Text (RNN + CTC)")
st.write("1. Click **Record** and speak. 2. Click **Transcribe** to see text.")

col1, col2 = st.columns(2)

with col1:
    if st.button("🎙 Record (4 seconds)"):
        try:
            record_audio(filename=AUDIO_PATH, duration=4)
            st.success("Recording finished! Now click 'Transcribe'.")
        except Exception as e:
            st.error(f"Recording failed: {e}")

with col2:
    if st.button("🧠 Transcribe Audio"):
        if os.path.exists(AUDIO_PATH):
            try:
                text = transcribe(AUDIO_PATH)
                st.subheader("Transcription:")
                st.write(text if text.strip() != "" else "(Empty / could not decode)")
            except Exception as e:
                st.error(f"Transcription failed: {e}")
        else:
            st.warning("No recording found. Please record first.")

st.markdown("---")
st.caption("Demo project for Speech-to-Text with RNN + CTC + Streamlit UI.")
