# SignLingo: Real-Time ASL Sign Language Tutor

An interactive sign language learning app that uses computer vision to recognize **American Sign Language (ASL) gestures in real-time** and provide instant feedback to the user.

## Demo

> Live on Streamlit: https://signlingo-asl.streamlit.app/

## Features

- **Real-time gesture recognition**: recognizes dynamic ASL signs via webcam
- **Wrist-relative normalization**: ensures accuracy regardless of hand position or camera angle
- **Live visual feedback**: tells users whether their sign is correct in real time
- **Custom data pipeline**: built and labeled a dataset of dynamic gesture sequences
- **Optimized inference**: balanced latency and accuracy for interactive use

## Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch |
| Computer Vision | MediaPipe Holistic, OpenCV |
| Frontend / Deployment | Streamlit |
| Language | Python |
| Feature Engineering | Wrist-relative landmark normalization |

## How It Works

1. Webcam captures hand and body landmarks via **MediaPipe Holistic**
2. Landmarks are normalized relative to wrist position → removes translation variance
3. Normalized features are passed to a **PyTorch classifier** trained on custom gesture data
4. Prediction is displayed to the user with real-time visual feedback

## Getting Started

```bash
# Clone the repo
git clone https://github.com/arianna-y/SignLingo.git
cd SignLingo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure

```
SignLingo/
├── app.py                  # Streamlit frontend + webcam loop
├── model.py                # PyTorch model definition
├── train.py                # Training pipeline
├── data_collection.py      # Custom data collection tool
├── utils.py                # Normalization + preprocessing
└── data/                   # Labeled gesture sequences
```

## Dataset

Gesture sequences were collected using a custom-built data pipeline with real-time visual feedback loops. Each sequence captures dynamic hand and body landmark trajectories across multiple frames.

## Results

- Robust to varying hand positions and camera distances due to wrist-relative normalization
- Low-latency inference suitable for real-time interactive deployment
