"""
utils/preprocess.py
Load and preprocess the RAVDESS dataset.

RAVDESS filename format:
  03-01-06-01-02-01-12.wav
  Modality-Vocal-Emotion-Intensity-Statement-Repetition-Actor

Emotion codes:
  01=neutral, 02=calm, 03=happy, 04=sad,
  05=angry, 06=fearful, 07=disgust, 08=surprised
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from utils.feature_extraction import extract_features


EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

EMOTIONS = list(EMOTION_MAP.values())


def load_ravdess_dataset(data_dir='data'):
    """
    Load all RAVDESS audio files and extract features.

    Args:
        data_dir (str): Path to the folder containing Actor_* subdirectories

    Returns:
        X (np.ndarray): Features array, shape (n_samples, 180, 174, 1)
        y (np.ndarray): One-hot encoded labels, shape (n_samples, 8)
        label_encoder (LabelEncoder): Fitted encoder
    """
    features = []
    labels = []

    # Find all wav files
    wav_files = glob.glob(os.path.join(data_dir, 'Actor_*', '*.wav'))

    if len(wav_files) == 0:
        print(f"⚠️  No .wav files found in '{data_dir}/Actor_*/'")
        print("    Please download RAVDESS from: https://zenodo.org/record/1188976")
        print("    and extract into the 'data/' folder.")
        return None, None, None

    print(f"📂 Found {len(wav_files)} audio files. Extracting features...")

    for file_path in tqdm(wav_files, desc="Extracting"):
        # Parse emotion from filename
        filename = os.path.basename(file_path)
        parts = filename.split('-')
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        if emotion_code not in EMOTION_MAP:
            continue

        emotion_label = EMOTION_MAP[emotion_code]

        # Extract features
        feat = extract_features(file_path)
        if feat is not None:
            features.append(feat)
            labels.append(emotion_label)

    if len(features) == 0:
        print("No features extracted. Check your data directory.")
        return None, None, None

    # Convert to numpy arrays
    X = np.array(features)
    X = X[..., np.newaxis]  # Add channel dim: (n, 180, 174, 1)

    # Encode labels
    le = LabelEncoder()
    le.fit(EMOTIONS)
    y_encoded = le.transform(labels)
    y = to_categorical(y_encoded, num_classes=len(EMOTIONS))

    print(f"\n✅ Dataset loaded: {X.shape[0]} samples, {len(EMOTIONS)} emotions")
    print(f"   Feature shape: {X.shape}")

    return X, y, le


def get_train_test_split(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split into train/validation/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=random_state
    )

    print(f"\n📊 Split sizes:")
    print(f"   Train:      {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test:       {X_test.shape[0]} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test
