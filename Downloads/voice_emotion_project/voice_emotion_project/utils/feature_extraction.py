"""
utils/feature_extraction.py
Extracts MFCC, Chroma, and Mel Spectrogram features from audio files.
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


def extract_features(file_path, max_pad_len=174):
    """
    Extract MFCC + Chroma + Mel Spectrogram from an audio file.

    Args:
        file_path (str): Path to .wav audio file
        max_pad_len (int): Fixed length for padding/truncating features

    Returns:
        np.ndarray: Feature array of shape (180, max_pad_len)
                    (40 MFCC + 12 Chroma + 128 Mel = 180 features)
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=3)

        # --- MFCC (40 coefficients) ---
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc = pad_or_truncate(mfcc, max_pad_len)

        # --- Chroma (12 pitch classes) ---
        stft = np.abs(librosa.stft(audio))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma = pad_or_truncate(chroma, max_pad_len)

        # --- Mel Spectrogram (128 bands) ---
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = pad_or_truncate(mel, max_pad_len)

        # Stack all features: shape (180, max_pad_len)
        features = np.vstack([mfcc, chroma, mel])
        return features

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None


def pad_or_truncate(feature, max_len):
    """Pad with zeros or truncate to fixed length."""
    if feature.shape[1] < max_len:
        pad_width = max_len - feature.shape[1]
        feature = np.pad(feature, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        feature = feature[:, :max_len]
    return feature


def extract_features_from_array(audio_array, sample_rate=22050, max_pad_len=174):
    """
    Extract features directly from a numpy audio array (for real-time use).

    Args:
        audio_array (np.ndarray): Raw audio signal
        sample_rate (int): Sample rate
        max_pad_len (int): Fixed length

    Returns:
        np.ndarray: Feature array of shape (1, 180, max_pad_len, 1)
    """
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=40)
    mfcc = pad_or_truncate(mfcc, max_pad_len)

    stft = np.abs(librosa.stft(audio_array))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    chroma = pad_or_truncate(chroma, max_pad_len)

    mel = librosa.feature.melspectrogram(y=audio_array, sr=sample_rate)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = pad_or_truncate(mel, max_pad_len)

    features = np.vstack([mfcc, chroma, mel])
    # Reshape for model input: (batch, height, width, channels)
    features = features[np.newaxis, ..., np.newaxis]
    return features
