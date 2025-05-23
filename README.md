# English Accent Detector from Video URL

## Overview
This is a Streamlit app that detects English accents (British, American, Australian, etc.) from a public video URL by:

1. Downloading the video from the provided URL (supports Loom, YouTube, direct MP4 links).
2. Extracting audio from the video.
3. Running an English accent classification model.
4. Displaying detected accent and confidence score.

## How to Run Locally

### Requirements
- Python 3.8+
- `ffmpeg` installed and added to your PATH (https://ffmpeg.org/download.html)

### Setup
```bash
pip install -r requirements.txt
