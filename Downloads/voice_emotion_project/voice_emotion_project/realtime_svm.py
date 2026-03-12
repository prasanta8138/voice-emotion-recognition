"""
realtime_svm.py
Real-time emotion detection using the SVM/MLP model.
Much more accurate than CNN+LSTM on small datasets.

Run:
    python realtime_svm.py
"""

import os
import pickle
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa

MODEL_PATH  = 'models/svm_model.pkl'
SAMPLE_RATE = 48000
DURATION    = 3
DEVICE      = 13  # Change if needed

EMOTION_COLORS = {
    'neutral':'#98989D', 'calm':'#5AC8FA', 'happy':'#FFD700',
    'sad':'#4A90D9', 'angry':'#FF3B30', 'fearful':'#BF5AF2',
    'disgust':'#FF9F0A', 'surprised':'#30D158'
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found! Run: python train_model_svm.py")
        exit(1)
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
    return data['model'], data['label_encoder']


def extract_features(audio, sr=22050):
    """Extract same features as training."""
    # Resample to 22050 if needed
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050

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
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    return np.array(features).reshape(1, -1)


def display_results(emotion, confidence, all_scores, audio):
    color = EMOTION_COLORS.get(emotion, '#00FF96')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#050A0E')

    # Waveform
    axes[0].set_facecolor('#060D0A')
    t = np.linspace(0, DURATION, len(audio))
    axes[0].plot(t, audio, color=color, linewidth=0.8)
    axes[0].fill_between(t, audio, alpha=0.1, color=color)
    axes[0].set_title(f'Emotion: {emotion.upper()}  |  {confidence*100:.1f}%',
                      color=color, fontsize=12)
    axes[0].set_xlabel('Time (s)', color='#88AA88')
    axes[0].tick_params(colors='#88AA88')
    for s in axes[0].spines.values(): s.set_color('#0F2A1A')

    # Bars
    axes[1].set_facecolor('#060D0A')
    emotions = list(all_scores.keys())
    scores = [all_scores[e]*100 for e in emotions]
    colors = [EMOTION_COLORS.get(e, '#00FF96') for e in emotions]
    bars = axes[1].barh(emotions, scores, color=colors, alpha=0.8)
    bars[emotions.index(emotion)].set_alpha(1.0)
    axes[1].set_xlim(0, 110)
    axes[1].set_title('Confidence Scores', color='#E0E8E0', fontsize=11)
    axes[1].set_xlabel('Confidence (%)', color='#88AA88')
    axes[1].tick_params(colors='#88AA88')
    for s in axes[1].spines.values(): s.set_color('#0F2A1A')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('results/plots/last_prediction.png',
                dpi=150, bbox_inches='tight', facecolor='#050A0E')
    plt.show()


def run():
    model, le = load_model()

    print("=" * 50)
    print("  🎙️  Real-Time Emotion Detection (SVM)")
    print("  Press ENTER to record | 'q' to quit")
    print("=" * 50)

    while True:
        user_input = input("\n> Press ENTER to record (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            print("👋 Exiting.")
            break

        try:
            print(f"🎙️  Recording for {DURATION} seconds... ", end='', flush=True)
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1, dtype='float32',
                           device=DEVICE)
            sd.wait()
            audio = audio.flatten()
            print("Done!")

            print("🔍 Analyzing...")
            features = extract_features(audio, SAMPLE_RATE)
            proba = model.predict_proba(features)[0]
            scores = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
            emotion = max(scores, key=scores.get)
            confidence = scores[emotion]

            print(f"\n{'='*40}")
            print(f"  Detected: {emotion.upper()}")
            print(f"  Confidence: {confidence*100:.1f}%")
            print(f"{'='*40}")
            for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(s * 30)
                print(f"  {e:<12} {bar:<30} {s*100:.1f}%")

            display_results(emotion, confidence, scores, audio)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    run()
