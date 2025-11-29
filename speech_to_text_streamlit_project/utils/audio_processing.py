import librosa
import numpy as np
import pandas as pd
import os

def extract_features(audio_path, sr=16000, n_mfcc=13):
    """Load audio file and compute MFCC features (T, n_mfcc)."""
    audio, sr = librosa.load(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (time_steps, n_mfcc)

def load_dataset(data_dir):
    """Load dataset described by metadata.csv and audio/ folder."""
    meta_path = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(meta_path)
    X, y, filepaths = [], [], []
    for _, row in df.iterrows():
        wav_path = os.path.join(data_dir, "audio", row["filename"])
        if not os.path.exists(wav_path):
            continue
        feat = extract_features(wav_path)
        X.append(feat)
        y.append(str(row["transcription"]).lower())
        filepaths.append(wav_path)
    return X, y, filepaths
