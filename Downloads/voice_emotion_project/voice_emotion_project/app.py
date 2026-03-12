"""
app.py
Full GUI Dashboard — Voice Emotion Recognition
Real-time waveform + emotion display with Tkinter + Matplotlib.

Run:
    python app.py
"""

import os
import sys
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, font
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import sounddevice as sd
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = 'models/emotion_model.h5'
SAMPLE_RATE = 22050
DURATION    = 3
CHUNK       = 1024

EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
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

BG      = '#050A0E'
PANEL   = '#060D0A'
BORDER  = '#0F2A1A'
GREEN   = '#00FF96'
DIMTEXT = '#88AA88'


class VoiceEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Voice Emotion Recognition — CNN+LSTM")
        self.root.configure(bg=BG)
        self.root.geometry("1100x720")
        self.root.resizable(True, True)

        self.model = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=SAMPLE_RATE * DURATION)
        self.current_emotion = tk.StringVar(value="—")
        self.current_confidence = tk.StringVar(value="—")
        self.status_text = tk.StringVar(value="STANDBY")
        self.detection_log = []

        self._load_model_async()
        self._build_ui()

    def _load_model_async(self):
        def load():
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.status_text.set("MODEL READY")
            except Exception as e:
                self.status_text.set(f"MODEL ERROR: {e}")
        threading.Thread(target=load, daemon=True).start()

    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self.root, bg=BG, pady=10)
        header.pack(fill='x', padx=20)

        tk.Label(header, text="VOICE EMOTION RECOGNITION",
                 bg=BG, fg=GREEN,
                 font=('Courier', 11, 'bold'),
                 letterSpacing=4).pack(side='left')

        tk.Label(header, textvariable=self.status_text,
                 bg=BG, fg=DIMTEXT,
                 font=('Courier', 9)).pack(side='right')

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill='x', padx=20)

        # ── Main Layout ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=20, pady=10)

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True)

        right = tk.Frame(main, bg=BG, width=260)
        right.pack(side='right', fill='y', padx=(12, 0))
        right.pack_propagate(False)

        # ── Waveform Plot ──
        self._build_waveform(left)

        # ── Emotion Display ──
        self._build_emotion_display(left)

        # ── Confidence Bars ──
        self._build_confidence_bars(left)

        # ── Controls ──
        self._build_controls(left)

        # ── Right Panel ──
        self._build_right_panel(right)

    def _build_waveform(self, parent):
        frame = tk.Frame(parent, bg=PANEL, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill='x', pady=(0, 8))

        tk.Label(frame, text="WAVEFORM ANALYSIS",
                 bg=PANEL, fg=GREEN + '88',
                 font=('Courier', 8), pady=5).pack(anchor='w', padx=10)

        self.wave_fig, self.wave_ax = plt.subplots(
            figsize=(8, 1.6), facecolor=PANEL)
        self.wave_ax.set_facecolor(PANEL)
        self.wave_ax.set_ylim(-1, 1)
        for spine in self.wave_ax.spines.values():
            spine.set_color(BORDER)
        self.wave_ax.tick_params(colors=DIMTEXT, labelsize=7)
        self.wave_ax.axhline(0, color=BORDER, linewidth=0.5)

        self.wave_line, = self.wave_ax.plot([], [], color=GREEN, linewidth=0.8)
        self.wave_fig.tight_layout(pad=0.3)

        canvas = FigureCanvasTkAgg(self.wave_fig, master=frame)
        canvas.get_tk_widget().pack(fill='x', padx=5, pady=(0, 5))
        self.wave_canvas = canvas

    def _build_emotion_display(self, parent):
        self.emotion_frame = tk.Frame(
            parent, bg=PANEL,
            highlightbackground=BORDER, highlightthickness=1)
        self.emotion_frame.pack(fill='x', pady=(0, 8))

        inner = tk.Frame(self.emotion_frame, bg=PANEL, pady=12)
        inner.pack(fill='x', padx=16)

        tk.Label(inner, text="DETECTED EMOTION",
                 bg=PANEL, fg=DIMTEXT,
                 font=('Courier', 8)).pack(anchor='w')

        self.emotion_label = tk.Label(
            inner, textvariable=self.current_emotion,
            bg=PANEL, fg=GREEN,
            font=('Courier', 26, 'bold'))
        self.emotion_label.pack(anchor='w')

        self.conf_label = tk.Label(
            inner, textvariable=self.current_confidence,
            bg=PANEL, fg=DIMTEXT,
            font=('Courier', 11))
        self.conf_label.pack(anchor='w')

    def _build_confidence_bars(self, parent):
        frame = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill='x', pady=(0, 8))

        tk.Label(frame, text="CONFIDENCE DISTRIBUTION",
                 bg=PANEL, fg=GREEN + '88',
                 font=('Courier', 8), pady=5).pack(anchor='w', padx=10)

        self.bar_fig, self.bar_ax = plt.subplots(figsize=(8, 1.8), facecolor=PANEL)
        self.bar_ax.set_facecolor(PANEL)
        for spine in self.bar_ax.spines.values():
            spine.set_color(BORDER)
        self.bar_ax.tick_params(colors=DIMTEXT, labelsize=7)
        self.bar_ax.set_xlim(0, 1)
        self.bar_ax.set_yticks([])
        self.bar_fig.tight_layout(pad=0.3)

        # Initial bars
        self.bar_rects = self.bar_ax.barh(
            range(len(EMOTIONS)),
            [0.125] * len(EMOTIONS),
            color=[EMOTION_COLORS[e] for e in EMOTIONS],
            alpha=0.7, height=0.6
        )
        self.bar_ax.set_yticks(range(len(EMOTIONS)))
        self.bar_ax.set_yticklabels(
            [e[:4].upper() for e in EMOTIONS],
            color=DIMTEXT, fontsize=7, fontfamily='monospace')

        canvas = FigureCanvasTkAgg(self.bar_fig, master=frame)
        canvas.get_tk_widget().pack(fill='x', padx=5, pady=(0, 5))
        self.bar_canvas = canvas

    def _build_controls(self, parent):
        frame = tk.Frame(parent, bg=BG, pady=6)
        frame.pack(fill='x')

        self.record_btn = tk.Button(
            frame,
            text="▶  START RECORDING",
            bg=BG, fg=GREEN,
            font=('Courier', 11, 'bold'),
            relief='flat',
            bd=0,
            highlightbackground=GREEN,
            highlightthickness=1,
            padx=20, pady=8,
            cursor='hand2',
            command=self._toggle_recording
        )
        self.record_btn.pack(side='left', padx=(0, 12))

        tk.Label(frame,
                 text="CNN+LSTM  |  RAVDESS  |  8 EMOTIONS",
                 bg=BG, fg='#222',
                 font=('Courier', 8)).pack(side='right')

    def _build_right_panel(self, parent):
        # Detection Log
        tk.Label(parent, text="DETECTION LOG",
                 bg=BG, fg=GREEN + '88',
                 font=('Courier', 8), pady=6).pack(anchor='w')

        log_frame = tk.Frame(parent, bg=PANEL,
                             highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill='both', expand=True)

        self.log_text = tk.Text(
            log_frame, bg=PANEL, fg=DIMTEXT,
            font=('Courier', 8), relief='flat',
            bd=0, state='disabled', wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=8)

        # Acoustic features list
        tk.Label(parent, text="FEATURES USED",
                 bg=BG, fg=GREEN + '88',
                 font=('Courier', 8), pady=6).pack(anchor='w')

        feat_frame = tk.Frame(parent, bg=PANEL,
                              highlightbackground=BORDER, highlightthickness=1)
        feat_frame.pack(fill='x')

        features = ['MFCC (40)', 'Chroma (12)', 'Mel Spec (128)',
                    'Pitch', 'Intensity', 'Prosody']
        for f in features:
            tk.Label(feat_frame, text=f"  {f}",
                     bg=PANEL, fg='#444',
                     font=('Courier', 8), pady=3).pack(anchor='w')

    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self.is_recording = True
        self.record_btn.config(text="■  STOP", fg='#FF3B30',
                               highlightbackground='#FF3B30')
        self.status_text.set("● RECORDING")
        self.current_emotion.set("...")
        self.current_confidence.set("Analyzing...")

        def record_thread():
            try:
                audio = sd.rec(
                    int(DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1, dtype='float32'
                )
                sd.wait()
                audio = audio.flatten()
                self._update_waveform(audio)
                self._predict(audio)
            except Exception as e:
                self.status_text.set(f"Error: {e}")
            finally:
                self._stop_recording()

        threading.Thread(target=record_thread, daemon=True).start()

    def _stop_recording(self):
        self.is_recording = False
        self.record_btn.config(text="▶  START RECORDING", fg=GREEN,
                               highlightbackground=GREEN)
        self.status_text.set("STANDBY")

    def _update_waveform(self, audio):
        x = np.linspace(0, DURATION, len(audio))
        self.wave_line.set_data(x, audio)
        self.wave_ax.set_xlim(0, DURATION)
        self.wave_ax.set_ylim(audio.min() * 1.1, audio.max() * 1.1)
        self.wave_canvas.draw()

    def _predict(self, audio):
        if self.model is None:
            self.status_text.set("Model not loaded!")
            return

        try:
            from utils.feature_extraction import extract_features_from_array
            features = extract_features_from_array(audio, SAMPLE_RATE)
            preds = self.model.predict(features, verbose=0)[0]

            scores = {e: float(p) for e, p in zip(EMOTIONS, preds)}
            top_emotion = max(scores, key=scores.get)
            confidence = scores[top_emotion]

            color = EMOTION_COLORS.get(top_emotion, GREEN)

            self.current_emotion.set(top_emotion.upper())
            self.current_confidence.set(f"Confidence: {confidence*100:.1f}%")
            self.emotion_label.config(fg=color)
            self.emotion_frame.config(highlightbackground=color + '44')

            self._update_bars(scores, top_emotion)
            self._add_log_entry(top_emotion, confidence)
            self.status_text.set("● DETECTION COMPLETE")

        except Exception as e:
            self.status_text.set(f"Predict error: {e}")

    def _update_bars(self, scores, top_emotion):
        for rect, emo in zip(self.bar_rects, EMOTIONS):
            val = scores.get(emo, 0)
            rect.set_width(val)
            alpha = 1.0 if emo == top_emotion else 0.4
            rect.set_alpha(alpha)
        self.bar_canvas.draw()

    def _add_log_entry(self, emotion, confidence):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        color = EMOTION_COLORS.get(emotion, GREEN)

        self.log_text.config(state='normal')
        self.log_text.insert('1.0', f"{ts}  {emotion.upper():<12} {confidence*100:.1f}%\n")
        self.log_text.config(state='disabled')


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model not found at '{MODEL_PATH}'")
        print("   The GUI will launch but detection won't work until you train.")
        print("   Run: python train_model.py\n")

    root = tk.Tk()
    app = VoiceEmotionApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
