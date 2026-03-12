"""
app_ui.py - NEUROVOX Voice Emotion Recognition Dashboard
Auto mic detection - works with both PC mic and earphone mic
Run: python app_ui.py
"""

import os
import pickle
import threading
import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import librosa
import warnings
import datetime
warnings.filterwarnings('ignore')

# ── CONFIG ──
MODEL_PATH = 'models/fixed_model.pkl'
DURATION   = 4
TARGET_SR  = 22050
BOOST      = 50

EMOTION_DATA = {
    'angry':    {'color': '#FF3B30', 'desc': 'HIGH ENERGY · NEGATIVE'},
    'calm':     {'color': '#5AC8FA', 'desc': 'LOW ENERGY · POSITIVE'},
    'happy':    {'color': '#FFD700', 'desc': 'HIGH ENERGY · POSITIVE'},
    'sad':      {'color': '#4A90D9', 'desc': 'LOW ENERGY · NEGATIVE'},
    'fearful':  {'color': '#BF5AF2', 'desc': 'HIGH AROUSAL · NEGATIVE'},
    'neutral':  {'color': '#98989D', 'desc': 'BASELINE · NEUTRAL'},
    'disgust':  {'color': '#FF9F0A', 'desc': 'AVERSIVE · NEGATIVE'},
    'surprised':{'color': '#30D158', 'desc': 'HIGH AROUSAL · MIXED'},
}

BG      = '#03080C'
PANEL   = '#060F14'
BORDER  = '#0A2030'
ACCENT  = '#00E5FF'
DIM     = '#1A3A4A'
DIMTEXT = '#4A7A8A'


def get_best_mic():
    """Auto detect best available input device."""
    # Try default device first
    try:
        default = sd.query_devices(kind='input')
        return default['index'], int(default['default_samplerate']), default['name']
    except Exception:
        pass
    # Fallback: scan all devices
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            return i, int(dev['default_samplerate']), dev['name']
    return 0, 44100, 'Unknown'


class VoiceEmotionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NEUROVOX — Voice Emotion Intelligence")
        self.root.configure(bg=BG)
        self.root.geometry("1200x760")

        self.model         = None
        self.le            = None
        self.is_recording  = False
        self.audio_data    = np.zeros(1000)
        self.phase         = 'idle'
        self.pulse_running = True

        # Auto detect mic
        self.device_idx, self.sample_rate, self.device_name = get_best_mic()

        self._load_model_async()
        self._build_ui()
        self._start_pulse()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.pulse_running = False
        self.root.destroy()

    def _load_model_async(self):
        def load():
            try:
                with open(MODEL_PATH, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.le    = data['label_encoder']
                self._set_status("SYSTEM READY", ACCENT)
            except Exception:
                self._set_status("MODEL ERROR", '#FF3B30')
        threading.Thread(target=load, daemon=True).start()

    def _build_ui(self):
        # Top bar
        topbar = tk.Frame(self.root, bg=BG, height=56)
        topbar.pack(fill='x')
        topbar.pack_propagate(False)

        tk.Label(topbar, text="NEUROVOX",
                 bg=BG, fg=ACCENT,
                 font=('Courier', 22, 'bold')).pack(side='left', padx=24, pady=10)

        tk.Label(topbar, text="VOICE EMOTION INTELLIGENCE SYSTEM",
                 bg=BG, fg=DIMTEXT,
                 font=('Courier', 9)).pack(side='left', pady=16)

        status_frame = tk.Frame(topbar, bg=BG)
        status_frame.pack(side='right', padx=24)
        self.status_dot = tk.Label(status_frame, text='●',
                                    bg=BG, fg='#333', font=('Courier', 12))
        self.status_dot.pack(side='left')
        self.status_label = tk.Label(status_frame, text='LOADING...',
                                      bg=BG, fg=DIMTEXT, font=('Courier', 9))
        self.status_label.pack(side='left', padx=4)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x')

        # Mic info bar
        mic_bar = tk.Frame(self.root, bg='#030D12', height=28)
        mic_bar.pack(fill='x')
        mic_bar.pack_propagate(False)
        mic_name = self.device_name[:50]
        tk.Label(mic_bar,
                 text=f"MIC: {mic_name}  |  SR: {self.sample_rate} Hz  |  AUTO DETECTED",
                 bg='#030D12', fg=ACCENT, font=('Courier', 8)).pack(side='left', padx=16, pady=5)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x')

        # Main
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill='both', expand=True, padx=16, pady=12)

        left = tk.Frame(main, bg=BG)
        left.pack(side='left', fill='both', expand=True)

        right = tk.Frame(main, bg=BG, width=300)
        right.pack(side='right', fill='y', padx=(12, 0))
        right.pack_propagate(False)

        self._build_waveform(left)
        self._build_emotion_display(left)
        self._build_controls(left)
        self._build_right_panel(right)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill='x')
        bottom = tk.Frame(self.root, bg=BG, height=28)
        bottom.pack(fill='x')
        tk.Label(bottom,
                 text="SVM RBF · BALANCED · RAVDESS + TESS · 4,240 SAMPLES · ACCURACY: 90.57%",
                 bg=BG, fg=DIM, font=('Courier', 8)).pack(side='left', padx=16, pady=6)

    def _build_waveform(self, parent):
        frame = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill='x', pady=(0, 10))
        header = tk.Frame(frame, bg=PANEL)
        header.pack(fill='x', padx=12, pady=(8, 0))
        tk.Label(header, text="ACOUSTIC WAVEFORM",
                 bg=PANEL, fg=DIMTEXT, font=('Courier', 8)).pack(side='left')
        self.wave_status = tk.Label(header, text="o IDLE",
                                     bg=PANEL, fg=DIMTEXT, font=('Courier', 8))
        self.wave_status.pack(side='right')

        self.wave_fig, self.wave_ax = plt.subplots(figsize=(9, 1.8), facecolor=PANEL)
        self.wave_ax.set_facecolor(PANEL)
        self.wave_ax.set_ylim(-1, 1)
        self.wave_ax.set_xlim(0, 1)
        for spine in self.wave_ax.spines.values():
            spine.set_color(BORDER)
        self.wave_ax.tick_params(colors=DIMTEXT, labelsize=7)
        self.wave_ax.axhline(0, color=DIM, linewidth=0.5, linestyle='--')
        self.wave_fig.tight_layout(pad=0.3)
        self.wave_line, = self.wave_ax.plot([], [], color=ACCENT, linewidth=1.2)
        canvas = FigureCanvasTkAgg(self.wave_fig, master=frame)
        canvas.get_tk_widget().pack(fill='x', padx=6, pady=(0, 8))
        self.wave_canvas = canvas

    def _build_emotion_display(self, parent):
        self.emo_frame = tk.Frame(parent, bg=PANEL,
                                   highlightbackground=BORDER, highlightthickness=1)
        self.emo_frame.pack(fill='x', pady=(0, 10))
        inner = tk.Frame(self.emo_frame, bg=PANEL)
        inner.pack(fill='x', padx=16, pady=14)

        left = tk.Frame(inner, bg=PANEL)
        left.pack(side='left', fill='both', expand=True)
        tk.Label(left, text="DETECTED EMOTION",
                 bg=PANEL, fg=DIMTEXT, font=('Courier', 8)).pack(anchor='w')
        self.emo_name = tk.Label(left, text="------",
                                  bg=PANEL, fg=ACCENT,
                                  font=('Courier', 32, 'bold'))
        self.emo_name.pack(anchor='w')
        self.emo_desc = tk.Label(left, text="AWAITING INPUT",
                                  bg=PANEL, fg=DIMTEXT, font=('Courier', 9))
        self.emo_desc.pack(anchor='w')

        right = tk.Frame(inner, bg=PANEL)
        right.pack(side='right', padx=16)
        tk.Label(right, text="CONFIDENCE",
                 bg=PANEL, fg=DIMTEXT, font=('Courier', 8)).pack(anchor='e')
        self.conf_label = tk.Label(right, text="---",
                                    bg=PANEL, fg=ACCENT,
                                    font=('Courier', 28, 'bold'))
        self.conf_label.pack(anchor='e')
        tk.Label(right, text="MIC LEVEL",
                 bg=PANEL, fg=DIMTEXT, font=('Courier', 7)).pack(anchor='e', pady=(8,0))
        self.mic_level = tk.Label(right, text="---",
                                   bg=PANEL, fg=DIMTEXT, font=('Courier', 9))
        self.mic_level.pack(anchor='e')

    def _build_controls(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill='x', pady=(0, 8))
        self.rec_btn = tk.Button(
            frame, text="  ANALYZE VOICE  ",
            bg=BG, fg=ACCENT,
            font=('Courier', 12, 'bold'),
            relief='flat', bd=0,
            highlightbackground=ACCENT, highlightthickness=2,
            padx=24, pady=10, cursor='hand2',
            activebackground='#001A22', activeforeground=ACCENT,
            command=self._toggle_recording
        )
        self.rec_btn.pack(side='left')

        info_frame = tk.Frame(frame, bg=BG)
        info_frame.pack(side='right')
        for label, val in [("ACCURACY","90.57%"),("DATASET","4,240"),("EMOTIONS","8")]:
            box = tk.Frame(info_frame, bg=PANEL,
                           highlightbackground=BORDER, highlightthickness=1)
            box.pack(side='left', padx=4, ipadx=10, ipady=6)
            tk.Label(box, text=val, bg=PANEL, fg=ACCENT,
                     font=('Courier', 12, 'bold')).pack()
            tk.Label(box, text=label, bg=PANEL, fg=DIMTEXT,
                     font=('Courier', 7)).pack()

    def _build_right_panel(self, parent):
        tk.Label(parent, text="EMOTION SCORES",
                 bg=BG, fg=DIMTEXT, font=('Courier', 8)).pack(anchor='w', pady=(0, 4))
        bars_frame = tk.Frame(parent, bg=PANEL,
                               highlightbackground=BORDER, highlightthickness=1)
        bars_frame.pack(fill='x')
        self.bar_widgets = {}
        for emo in sorted(EMOTION_DATA.keys()):
            row = tk.Frame(bars_frame, bg=PANEL)
            row.pack(fill='x', padx=8, pady=3)
            color = EMOTION_DATA[emo]['color']
            tk.Label(row, text=emo[:4].upper(),
                     bg=PANEL, fg=color,
                     font=('Courier', 8), width=5, anchor='w').pack(side='left')
            bar_bg = tk.Frame(row, bg=DIM, height=6)
            bar_bg.pack(side='left', fill='x', expand=True, pady=2)
            bar_fill = tk.Frame(bar_bg, bg=color, height=6)
            bar_fill.place(x=0, y=0, width=0, height=6)
            pct_label = tk.Label(row, text="0%",
                                  bg=PANEL, fg=DIMTEXT,
                                  font=('Courier', 7), width=4)
            pct_label.pack(side='left')
            self.bar_widgets[emo] = {
                'bg': bar_bg, 'fill': bar_fill,
                'label': pct_label, 'color': color
            }

        tk.Label(parent, text="DETECTION LOG",
                 bg=BG, fg=DIMTEXT, font=('Courier', 8)).pack(anchor='w', pady=(12, 4))
        log_frame = tk.Frame(parent, bg=PANEL,
                              highlightbackground=BORDER, highlightthickness=1)
        log_frame.pack(fill='both', expand=True)
        self.log_text = tk.Text(log_frame, bg=PANEL, fg=DIMTEXT,
                                 font=('Courier', 8), relief='flat',
                                 bd=0, state='disabled', wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=8, pady=8)

    def _set_status(self, text, color):
        try:
            self.status_label.config(text=text, fg=color)
            self.status_dot.config(fg=color)
        except Exception:
            pass

    def _start_pulse(self):
        def pulse():
            if not self.pulse_running:
                return
            try:
                if self.phase == 'recording':
                    self.wave_status.config(text="* RECORDING", fg='#FF3B30')
                elif self.phase == 'analyzing':
                    self.wave_status.config(text="* ANALYZING", fg='#FFD700')
                elif self.phase == 'result':
                    self.wave_status.config(text="* COMPLETE", fg='#30D158')
                else:
                    self.wave_status.config(text="o IDLE", fg=DIMTEXT)
            except Exception:
                pass
            self.root.after(100, pulse)
        pulse()

    def _toggle_recording(self):
        if not self.is_recording:
            # Re-detect mic every time (handles plug/unplug)
            self.device_idx, self.sample_rate, self.device_name = get_best_mic()
            self._start_recording()

    def _start_recording(self):
        if self.model is None:
            self._set_status("MODEL NOT LOADED", '#FF3B30')
            return

        self.is_recording = True
        self.phase = 'recording'
        self.rec_btn.config(text="  RECORDING...  ", fg='#FF3B30',
                             highlightbackground='#FF3B30', state='disabled')
        self._set_status("RECORDING — SPEAK NOW!", '#FF3B30')
        self.emo_name.config(text="...", fg=ACCENT)
        self.emo_desc.config(text="CAPTURING AUDIO...")
        self.conf_label.config(text="...")

        device_idx  = self.device_idx
        sample_rate = self.sample_rate

        def record_thread():
            try:
                audio = sd.rec(
                    int(DURATION * sample_rate),
                    samplerate=sample_rate,
                    channels=2,
                    dtype='float32',
                    device=device_idx
                )
                sd.wait()

                if audio.ndim > 1:
                    audio = audio[:, 0]
                else:
                    audio = audio.flatten()

                mic_amp = np.max(np.abs(audio))
                level_color = '#30D158' if mic_amp > 0.005 else '#FF9F0A'
                self.mic_level.config(text=f"{mic_amp:.4f}", fg=level_color)

                if mic_amp < 0.0001:
                    self._set_status("TOO QUIET! SPEAK LOUDER!", '#FF9F0A')
                    self.emo_name.config(text="QUIET", fg='#FF9F0A')
                    self.emo_desc.config(text="SPEAK LOUDER OR CHECK MIC")
                    return

                self.audio_data = audio
                self._update_waveform(audio, ACCENT)

                self.phase = 'analyzing'
                self._set_status("ANALYZING...", '#FFD700')
                self.emo_name.config(text="...", fg='#FFD700')
                self.emo_desc.config(text="PROCESSING FEATURES")

                emotion, confidence, scores = self._predict(audio, sample_rate)
                self._update_results(emotion, confidence, scores)

            except Exception as e:
                self._set_status("ERROR", '#FF3B30')
                self.emo_name.config(text="ERROR", fg='#FF3B30')
                self.emo_desc.config(text=str(e)[:60])
            finally:
                self.is_recording = False
                self.phase = 'result'
                self.rec_btn.config(
                    text="  ANALYZE VOICE  ",
                    fg=ACCENT, highlightbackground=ACCENT, state='normal'
                )

        threading.Thread(target=record_thread, daemon=True).start()

    def _predict(self, audio, sample_rate):
        audio_b = audio * BOOST
        if sample_rate != TARGET_SR:
            audio_b = librosa.resample(audio_b, orig_sr=sample_rate, target_sr=TARGET_SR)
        sr = TARGET_SR
        if np.max(np.abs(audio_b)) > 0:
            audio_b = audio_b / np.max(np.abs(audio_b))

        features = []
        mfcc = librosa.feature.mfcc(y=audio_b, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        stft = np.abs(librosa.stft(audio_b))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=audio_b, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))
        zcr = librosa.feature.zero_crossing_rate(audio_b)
        features.extend([np.mean(zcr), np.std(zcr)])
        rms = librosa.feature.rms(y=audio_b)
        features.extend([np.mean(rms), np.std(rms)])
        cent = librosa.feature.spectral_centroid(y=audio_b, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])
        rolloff = librosa.feature.spectral_rolloff(y=audio_b, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        X = np.array(features).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        scores = {self.le.classes_[i]: float(proba[i])
                  for i in range(len(self.le.classes_))}
        emotion = max(scores, key=scores.get)
        return emotion, scores[emotion], scores

    def _update_waveform(self, audio, color):
        try:
            t = np.linspace(0, 1, len(audio))
            boosted = audio * BOOST
            norm = np.max(np.abs(boosted)) + 1e-6
            self.wave_line.set_data(t, boosted / norm)
            self.wave_line.set_color(color)
            self.wave_ax.set_xlim(0, 1)
            self.wave_ax.set_ylim(-1.1, 1.1)
            self.wave_canvas.draw()
        except Exception:
            pass

    def _update_results(self, emotion, confidence, scores):
        data  = EMOTION_DATA.get(emotion, {})
        color = data.get('color', ACCENT)
        self.emo_name.config(text=emotion.upper(), fg=color)
        self.emo_desc.config(text=data.get('desc', ''), fg=color)
        self.conf_label.config(text=f"{confidence*100:.1f}%", fg=color)
        self.emo_frame.config(highlightbackground=color)
        self._set_status("DETECTION COMPLETE", '#30D158')
        self._update_waveform(self.audio_data, color)
        self._update_bars(scores)
        self._add_log(emotion, confidence, color)

    def _update_bars(self, scores):
        self.root.update_idletasks()
        for emo, widgets in self.bar_widgets.items():
            score = scores.get(emo, 0)
            color = widgets['color']
            bg_w  = widgets['bg'].winfo_width() or 150
            fill_w = max(2, int(bg_w * score))
            widgets['fill'].place(x=0, y=0, width=fill_w, height=6)
            widgets['label'].config(
                text=f"{int(score*100)}%",
                fg=color if score > 0.1 else DIMTEXT
            )

    def _add_log(self, emotion, confidence, color):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"{ts}  {emotion.upper():<12} {confidence*100:.0f}%\n"
        self.log_text.config(state='normal')
        self.log_text.insert('1.0', entry)
        self.log_text.config(state='disabled')


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found! Run: python train_fixed.py first")
    root = tk.Tk()
    app = VoiceEmotionUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
