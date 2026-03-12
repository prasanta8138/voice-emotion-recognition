"""
debug_mic.py
Debug microphone capture and feature extraction.
Run: python debug_mic.py
"""

import numpy as np
import sounddevice as sd
import librosa
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 48000
DURATION    = 3
DEVICE      = 19

print("Recording 3 seconds of SILENCE...")
audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=2, dtype='float32', device=DEVICE)
sd.wait()
print(f"Audio shape: {audio.shape}")
audio_mono = audio[:, 0]
print(f"Max amplitude (silence): {np.max(np.abs(audio_mono)):.6f}")
print(f"Mean amplitude (silence): {np.mean(np.abs(audio_mono)):.6f}")

print("\nNow speak loudly for 3 seconds...")
input("Press ENTER to start recording...")
audio2 = sd.rec(int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=2, dtype='float32', device=DEVICE)
sd.wait()
audio2_mono = audio2[:, 0]
print(f"Max amplitude (speech): {np.max(np.abs(audio2_mono)):.6f}")
print(f"Mean amplitude (speech): {np.mean(np.abs(audio2_mono)):.6f}")

if np.max(np.abs(audio2_mono)) > np.max(np.abs(audio_mono)) * 2:
    print("\n[OK] Microphone is capturing voice correctly!")
else:
    print("\n[!!] Microphone NOT capturing voice properly!")
    print("     Try a different device number.")
