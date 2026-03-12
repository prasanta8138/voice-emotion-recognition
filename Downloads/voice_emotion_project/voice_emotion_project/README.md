# NEUROVOX — Voice Emotion Recognition System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-90.57%25-green.svg)
![Emotions](https://img.shields.io/badge/Emotions-8-orange.svg)
![Dataset](https://img.shields.io/badge/Samples-4240-purple.svg)

A real-time Speech Emotion Recognition (SER) system that detects 8 human emotions from voice using Machine Learning.

---

## Detected Emotions
| Emotion | Color |
|---------|-------|
| Angry | Red |
| Calm | Blue |
| Disgust | Orange |
| Fearful | Purple |
| Happy | Yellow |
| Neutral | Grey |
| Sad | Blue |
| Surprised | Green |

---

## Model Performance
| Model | Accuracy |
|-------|----------|
| SVM (RBF, Balanced) | **90.57%** |
| MLP (512-256-128) | 88.80% |

### Per-Emotion Accuracy
| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|----|
| Angry | 0.87 | 0.97 | 0.92 |
| Calm | 0.81 | 0.92 | 0.86 |
| Disgust | 0.97 | 0.81 | 0.88 |
| Fearful | 0.93 | 0.95 | 0.94 |
| Happy | 0.94 | 0.86 | 0.89 |
| Neutral | 0.90 | 0.96 | 0.93 |
| Sad | 0.86 | 0.90 | 0.88 |
| Surprised | 0.94 | 0.90 | 0.92 |

---

## Datasets
| Dataset | Samples | Emotions |
|---------|---------|----------|
| RAVDESS | 1,440 | 8 |
| TESS | 2,800 | 7 |
| **Total** | **4,240** | **8** |

---

## Features Extracted
- MFCC (40 coefficients) mean + std
- Chroma STFT (12) mean + std
- Mel Spectrogram (128) mean + std
- Zero Crossing Rate
- RMS Energy
- Spectral Centroid
- Spectral Rolloff

**Total: 368 features per audio sample**

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/voice-emotion-recognition.git
cd voice-emotion-recognition
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

```bash
# Train the model
python train_fixed.py

# Real-time detection terminal
python realtime_fixed.py

# Professional UI dashboard
python app_ui.py
```

---

## Project Structure
```
voice_emotion_project/
├── data/                  # RAVDESS dataset
├── data_tess/             # TESS dataset
├── models/                # Saved models
├── results/plots/         # Charts
├── train_fixed.py         # Training script
├── realtime_fixed.py      # Terminal detection
├── app_ui.py              # UI dashboard
└── requirements.txt
```

---

## How It Works
1. Record 4 seconds of audio via microphone
2. Extract 368 acoustic features using librosa
3. Predict using SVM RBF classifier with class balancing
4. Display real-time emotion with confidence score

---

## Final Year AI Project
Built as a Final Year Project demonstrating speech signal processing, machine learning classification, real-time audio analysis, and professional GUI development.
