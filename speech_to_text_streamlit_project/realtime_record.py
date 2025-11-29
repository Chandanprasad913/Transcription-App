import sounddevice as sd
import soundfile as sf

def record_audio(filename="live.wav", duration=4, sr=16000, device=None):
    """Record from default microphone and save to WAV."""
    print("🎙 Recording... Speak now...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, device=device)
    sd.wait()
    sf.write(filename, audio, sr)
    print(f"✔ Saved recording to {filename}")

if __name__ == "__main__":
    record_audio()
