"""
realtime_combined.py
Real-time Voice Emotion Detection
RAVDESS + TESS | SVM | 90.68% Accuracy

Run:
    python realtime_combined.py
"""

import os
import pickle
import numpy as np
import sounddevice as sd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import librosa
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──
MODEL_PATH  = 'models/combined_model.pkl'
SAMPLE_RATE = 44100
DURATION    = 4
DEVICE      = 19
TARGET_SR   = 22050

EMOTION_COLORS = {
    'neutral':   '#98989D',
    'calm':      '#5AC8FA',
    'happy':     '#FFD700',
    'sad':       '#4A90D9',
    'angry':     '#FF3B30',
    'fearful':   '#BF5AF2',
    'disgust':   '#FF9F0A',
    'surprised': '#30D158'
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found! Run: python train_combined.py first")
        exit(1)
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    print("[OK] Model loaded!")
    return data['model'], data['label_encoder']


def extract_features(audio, orig_sr):
    # Resample to match training
    if orig_sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=TARGET_SR)
    sr = TARGET_SR

    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    features = []

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))

    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([np.mean(zcr), np.std(zcr)])

    rms = librosa.feature.rms(y=audio)
    features.extend([np.mean(rms), np.std(rms)])

    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.extend([np.mean(cent), np.std(cent)])

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.extend([np.mean(rolloff), np.std(rolloff)])

    return np.array(features).reshape(1, -1)


def display_results(emotion, confidence, scores, audio):
    color = EMOTION_COLORS.get(emotion, '#00FF96')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#050A0E')
    fig.suptitle('Voice Emotion Recognition  |  RAVDESS + TESS  |  90.68%',
                 color='#00E5FF44', fontsize=10)

    # Waveform
    axes[0].set_facecolor('#060D0A')
    t = np.linspace(0, DURATION, len(audio))
    axes[0].plot(t, audio, color=color, linewidth=0.8)
    axes[0].fill_between(t, audio, alpha=0.1, color=color)
    axes[0].set_title(f'{emotion.upper()}  |  {confidence*100:.1f}%',
                      color=color, fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (s)', color='#88AA88')
    axes[0].set_ylabel('Amplitude', color='#88AA88')
    axes[0].tick_params(colors='#88AA88')
    axes[0].grid(True, color='#0F2A1A', alpha=0.4)
    for s in axes[0].spines.values():
        s.set_color('#0F2A1A')

    # Confidence bars
    axes[1].set_facecolor('#060D0A')
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    emos   = [e for e, _ in sorted_scores]
    vals   = [v * 100 for _, v in sorted_scores]
    colors = [EMOTION_COLORS.get(e, '#00FF96') for e in emos]

    bars = axes[1].barh(emos, vals, color=colors, alpha=0.7, height=0.6)
    bars[0].set_alpha(1.0)
    bars[0].set_edgecolor(colors[0])
    bars[0].set_linewidth(2)

    for bar, val in zip(bars, vals):
        axes[1].text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{val:.1f}%', va='center',
                     color='#E0E8E0', fontsize=9)

    axes[1].set_xlim(0, 115)
    axes[1].set_title('All Emotion Scores', color='#E0E8E0', fontsize=11)
    axes[1].set_xlabel('Confidence (%)', color='#88AA88')
    axes[1].tick_params(colors='#88AA88')
    axes[1].invert_yaxis()
    for s in axes[1].spines.values():
        s.set_color('#0F2A1A')

    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/last_prediction.png',
                dpi=150, bbox_inches='tight', facecolor='#050A0E')
    plt.show()
    plt.close()


def run():
    model, le = load_model()

    print("=" * 52)
    print("  Real-Time Voice Emotion Detection")
    print("  Model: RAVDESS + TESS | Accuracy: 90.68%")
    print("  Press ENTER to record | 'q' to quit")
    print("=" * 52)
    print()
    print("  Tips for best results:")
    print("  Angry    -- speak LOUD and fast")
    print("  Happy    -- speak cheerfully")
    print("  Sad      -- speak slowly and softly")
    print("  Fearful  -- speak with trembling voice")
    print("  Neutral  -- speak normally")
    print()

    while True:
        user_input = input("> Press ENTER to record (or 'q' to quit): ").strip().lower()

        if user_input == 'q':
            print("Exiting.")
            break

        try:
            print(f"Recording {DURATION} seconds... speak now!", end='', flush=True)
            audio = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=2,
                dtype='float32',
                device=DEVICE
            )
            sd.wait()

            # Take first channel
            if audio.ndim > 1:
                audio = audio[:, 0]
            else:
                audio = audio.flatten()

            print(" Done!")
            print("Analyzing emotion...")

            features   = extract_features(audio, SAMPLE_RATE)
            proba      = model.predict_proba(features)[0]
            scores     = {le.classes_[i]: float(proba[i])
                          for i in range(len(le.classes_))}
            emotion    = max(scores, key=scores.get)
            confidence = scores[emotion]

            print()
            print("=" * 42)
            print(f"  Detected  : {emotion.upper()}")
            print(f"  Confidence: {confidence*100:.1f}%")
            print("=" * 42)

            for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                bar = '#' * int(s * 30)
                print(f"  {e:<12} {bar:<30} {s*100:.1f}%")

            print()
            print("Opening visualization...")
            display_results(emotion, confidence, scores, audio)

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    run()
