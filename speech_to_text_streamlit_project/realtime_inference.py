import os
import numpy as np
import tensorflow as tf
from utils.audio_processing import extract_features
from models.rnn_ctc_model import int_sequence_to_text, VOCAB_SIZE

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "rnn_ctc_inference_model.h5")

def greedy_decode(probs):
    """Greedy CTC decode: argmax then remove duplicates and blanks."""
    best_path = np.argmax(probs, axis=-1)
    decoded = []
    prev = VOCAB_SIZE - 1  # blank id
    for p in best_path:
        if p != prev and p != (VOCAB_SIZE - 1):
            decoded.append(p)
        prev = p
    return decoded

def transcribe(audio_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Inference model not found. Train first via main_train.py")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    feats = extract_features(audio_path)
    x = np.expand_dims(feats, axis=0)
    probs = model.predict(x)[0]
    seq = greedy_decode(probs)
    text = int_sequence_to_text(seq)
    return text

if __name__ == "__main__":
    test_path = os.path.join(BASE_DIR, "data", "audio", "hello.wav")
    print("Transcription:", transcribe(test_path))
