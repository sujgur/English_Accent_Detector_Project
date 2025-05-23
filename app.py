import os
import subprocess
import streamlit as st
import librosa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# --- Setup model once ---
@st.cache_resource(show_spinner=False)
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("dima806/english_accents_classification")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("dima806/english_accents_classification")
    return processor, model

processor, model = load_model()

def download_video(url, output_path="video.mp4"):
    command = ["yt-dlp", "-o", output_path, url]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Failed to download video: {result.stderr.decode()}")
    return output_path

def extract_audio(video_path, audio_path="audio.wav"):
    command = [
        "ffmpeg",
        "-y",              # overwrite if exists
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise Exception(f"Failed to extract audio: {result.stderr.decode()}")
    return audio_path

def predict_accent(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = torch.argmax(logits, dim=-1).item()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    confidence = float(probs[predicted_id]) * 100
    label = model.config.id2label[predicted_id]
    return label, confidence

# --- Streamlit UI ---
st.title("🎙️ English Accent Detection from Video URL")

video_url = st.text_input("Enter a public video URL (Loom, direct MP4, YouTube, etc):")

if st.button("Analyze Accent") and video_url.strip():
    try:
        with st.spinner("Downloading video..."):
            video_path = download_video(video_url)

        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path)

        with st.spinner("Analyzing accent..."):
            label, confidence = predict_accent(audio_path)

        st.success(f"Accent Detected: **{label}**")
        st.write(f"Confidence Score: **{confidence:.2f}%**")
        st.write("**Explanation:** The model analyzed the speaker's audio characteristics and matched them to typical English accents.")

    except Exception as e:
        st.error(f"Error: {e}")

    finally:
        # Clean up files if they exist
        if os.path.exists("video.mp4"):
            os.remove("video.mp4")
        if os.path.exists("audio.wav"):
            os.remove("audio.wav")
