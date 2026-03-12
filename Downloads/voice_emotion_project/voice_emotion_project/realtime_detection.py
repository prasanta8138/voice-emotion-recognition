"""
realtime_detection.py
Real-time microphone emotion detection using a trained CNN+LSTM model.

Run:
    python realtime_detection.py

Controls:
    Press ENTER to record a 3-second audio clip.
    Type 'q' to quit.
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from utils.feature_extraction import extract_features_from_array
from utils.preprocess import EMOTIONS


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH   = 'models/emotion_model.h5'
SAMPLE_RATE  = 48000
DURATION     = 3       # seconds to record
CHANNELS     = 1

EMOTION_COLORS = {
    'neutral':   '#98989D',
    'calm':      '#5AC8FA',
    'happy':     '#FFD700',
    'sad':       '#4A90D9',
    'angry':     '#FF3B30',
    'fearful':   '#BF5AF2',
    'disgust':   '#FF9F0A',
    'surprised': '#30D158',
}


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at '{MODEL_PATH}'")
        print("   Run 'python train_model.py' first.")
        sys.exit(1)

    print(f"✅ Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("   Model loaded.\n")
    return model


def record_audio(duration=DURATION, sample_rate=SAMPLE_RATE):
    """Record audio from the microphone."""
    print(f"🎙️  Recording for {duration} seconds... ", end='', flush=True)
    audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype='float32',
    device=13
)
    sd.wait()
    audio = audio.flatten()
    print("Done!")
    return audio, sample_rate


def predict_emotion(model, audio, sample_rate):
    """Extract features and predict emotion from audio."""
    features = extract_features_from_array(audio, sample_rate)
    predictions = model.predict(features, verbose=0)[0]

    results = {
        emotion: float(conf)
        for emotion, conf in zip(EMOTIONS, predictions)
    }
    top_emotion = max(results, key=results.get)
    top_confidence = results[top_emotion]

    return top_emotion, top_confidence, results


def display_results(emotion, confidence, all_scores, audio):
    """Display styled emotion results with waveform and bar chart."""
    color = EMOTION_COLORS.get(emotion, '#00FF96')

    fig = plt.figure(figsize=(14, 7), facecolor='#050A0E')
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_wave  = fig.add_subplot(gs[0, :])   # Waveform (top, full width)
    ax_bars  = fig.add_subplot(gs[1, 0])   # Confidence bars
    ax_radar = fig.add_subplot(gs[1, 1])   # Emotion scores text

    for ax in [ax_wave, ax_bars, ax_radar]:
        ax.set_facecolor('#060D0A')
        for spine in ax.spines.values():
            spine.set_color('#0F2A1A')
        ax.tick_params(colors='#88AA88', labelsize=9)

    # ── Waveform ──
    time_axis = np.linspace(0, DURATION, len(audio))
    ax_wave.plot(time_axis, audio, color=color, linewidth=0.8, alpha=0.9)
    ax_wave.axhline(0, color='#0F2A1A', linewidth=0.5)
    ax_wave.fill_between(time_axis, audio, alpha=0.1, color=color)
    ax_wave.set_title(
        f'Detected Emotion: {emotion.upper()}  •  Confidence: {confidence*100:.1f}%',
        color=color, fontsize=13, fontweight='bold', pad=10
    )
    ax_wave.set_xlabel('Time (s)', color='#88AA88')
    ax_wave.set_ylabel('Amplitude', color='#88AA88')
    ax_wave.grid(True, color='#0F2A1A', alpha=0.5, linewidth=0.5)

    # ── Confidence Bars ──
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    emotions_list = [e for e, _ in sorted_scores]
    scores_list   = [s * 100 for _, s in sorted_scores]
    bar_colors    = [EMOTION_COLORS.get(e, '#00FF96') for e in emotions_list]

    bars = ax_bars.barh(emotions_list, scores_list, color=bar_colors, alpha=0.8, height=0.6)
    # Highlight top bar
    bars[0].set_alpha(1.0)
    bars[0].set_edgecolor(bar_colors[0])
    bars[0].set_linewidth(1.5)

    for bar, score in zip(bars, scores_list):
        ax_bars.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{score:.1f}%', va='center', color='#E0E8E0', fontsize=8)

    ax_bars.set_xlim(0, 115)
    ax_bars.set_title('Emotion Confidence Scores', color='#E0E8E0', fontsize=10, pad=8)
    ax_bars.set_xlabel('Confidence (%)', color='#88AA88')
    ax_bars.grid(True, axis='x', color='#0F2A1A', alpha=0.5)
    ax_bars.invert_yaxis()

    # ── Score Table ──
    ax_radar.axis('off')
    ax_radar.set_title('Score Summary', color='#E0E8E0', fontsize=10, pad=8)
    y_pos = 0.95
    for emo, score in sorted_scores:
        c = EMOTION_COLORS.get(emo, '#00FF96')
        prefix = '▶ ' if emo == emotion else '  '
        ax_radar.text(0.05, y_pos, f"{prefix}{emo.capitalize():<12}", 
                      transform=ax_radar.transAxes,
                      color=c if emo == emotion else '#555',
                      fontsize=10, fontfamily='monospace',
                      fontweight='bold' if emo == emotion else 'normal')
        ax_radar.text(0.72, y_pos, f"{score*100:5.1f}%",
                      transform=ax_radar.transAxes,
                      color=c if emo == emotion else '#444',
                      fontsize=10, fontfamily='monospace')
        y_pos -= 0.11

    fig.suptitle('🎙️ Voice Emotion Recognition — CNN+LSTM on RAVDESS',
                 color='#00FF9688', fontsize=11, y=0.98)

    plt.savefig('results/plots/last_prediction.png',
                dpi=150, bbox_inches='tight', facecolor='#050A0E')
    plt.show()


def run():
    model = load_model()

    print("=" * 50)
    print("  🎙️  Real-Time Emotion Detection")
    print("  Press ENTER to record | Type 'q' to quit")
    print("=" * 50)

    while True:
        user_input = input("\n> Press ENTER to record (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            print("👋 Exiting.")
            break

        try:
            audio, sr = record_audio()
            print("🔍 Analyzing emotion...")

            emotion, confidence, all_scores = predict_emotion(model, audio, sr)

            print(f"\n{'='*40}")
            print(f"  Detected: {emotion.upper()}")
            print(f"  Confidence: {confidence*100:.1f}%")
            print(f"{'='*40}")
            print("\nAll scores:")
            for emo, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                bar = '█' * int(score * 30)
                print(f"  {emo:<12} {bar:<30} {score*100:.1f}%")

            print("\n📊 Opening visualization...")
            display_results(emotion, confidence, all_scores, audio)

        except KeyboardInterrupt:
            print("\n⏹️  Interrupted.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == '__main__':
    run()
