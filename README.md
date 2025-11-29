# Speech-to-Text RNN Demo with Streamlit

This project shows a minimal end-to-end Speech-to-Text pipeline:

- Extract MFCC features from audio (librosa)
- Train a small BiLSTM + CTC model (TensorFlow / Keras)
- Record microphone audio (sounddevice)
- Transcribe audio to text
- Simple UI built with Streamlit

## Structure

- `data/` — dummy WAV files + metadata
- `utils/audio_processing.py` — feature extraction and dataset loading
- `models/rnn_ctc_model.py` — BiLSTM + CTC model definition
- `main_train.py` — demo training script
- `realtime_record.py` — microphone recording helper
- `realtime_inference.py` — transcription helper
- `app.py` — Streamlit UI
- `requirements.txt` — dependencies

## Setup

```bash
pip install -r requirements.txt
```

> On Windows you may also need:
>
> - Latest audio drivers
> - Microphone permissions in Privacy settings

## Train the Model (Demo)

```bash
python main_train.py
```

This trains on the tiny dummy dataset and saves:

- `models/rnn_ctc_train_model.h5`
- `models/rnn_ctc_inference_model.h5`

For real use, replace `data/audio/*.wav` and `data/metadata.csv` with actual speech and transcripts.

## Run Streamlit UI

```bash
streamlit run app.py
```

Steps:

1. Click **"Record (4 seconds)"** and speak into your mic.
2. Click **"Transcribe Audio"** to see the decoded text.

### Notes

- This is a _demo_, not a production-grade ASR.
- Accuracy will be low on random speech unless you train on real data.
- To improve accuracy, train on more real speech data or switch to advanced models like Whisper or Wav2Vec2.
