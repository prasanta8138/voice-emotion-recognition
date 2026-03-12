"""
train_fixed.py
Fixed training with class balancing and better SVM params.
Run: python train_fixed.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
import pickle

RAVDESS_DIR = 'data'
TESS_DIR    = 'data_tess/TESS Toronto emotional speech set data'
MODEL_PATH  = 'models/fixed_model.pkl'
RESULTS_DIR = 'results/plots'

os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RAVDESS_MAP = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}

TESS_MAP = {
    'angry':'angry','disgust':'disgust','fear':'fearful',
    'happy':'happy','neutral':'neutral','sad':'sad',
    'pleasant_surprise':'surprised','pleasant_surprised':'surprised'
}


def extract_features(file_path, sr_target=22050):
    try:
        y, sr = librosa.load(file_path, duration=3, res_type='kaiser_fast')
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        features = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(cent), np.std(cent)])
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        return np.array(features)
    except Exception:
        return None


def load_ravdess():
    print("Loading RAVDESS...")
    X, y = [], []
    files = glob.glob(os.path.join(RAVDESS_DIR, 'Actor_*', '*.wav'))
    for f in tqdm(files, desc="  RAVDESS"):
        parts = os.path.basename(f).split('-')
        if len(parts) < 3: continue
        code = parts[2]
        if code not in RAVDESS_MAP: continue
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(RAVDESS_MAP[code])
    print(f"   RAVDESS: {len(X)} samples")
    return X, y


def load_tess():
    print("Loading TESS...")
    X, y = [], []
    for folder in glob.glob(os.path.join(TESS_DIR, '*')):
        fname = os.path.basename(folder).lower()
        emotion = None
        for key, val in TESS_MAP.items():
            if key in fname:
                emotion = val
                break
        if emotion is None: continue
        for f in glob.glob(os.path.join(folder, '*.wav')):
            feat = extract_features(f)
            if feat is not None:
                X.append(feat)
                y.append(emotion)
    print(f"   TESS: {len(X)} samples")
    return X, y


def train():
    print("=" * 55)
    print("  Voice Emotion Recognition — FIXED MODEL")
    print("  Class Balancing + Better SVM Parameters")
    print("=" * 55)

    X1, y1 = load_ravdess()
    X2, y2 = load_tess()

    X = np.array(X1 + X2)
    y_raw = y1 + y2

    print(f"\nTotal: {len(X)} samples")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Emotions: {list(le.classes_)}")

    # Show class distribution
    print("\nClass distribution:")
    for i, cls in enumerate(le.classes_):
        count = np.sum(y == i)
        print(f"  {cls:<12} {count} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # SVM with class_weight='balanced' to fix bias
    print("\nTraining SVM with class balancing...")
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            probability=True,
            class_weight='balanced',  # KEY FIX
            random_state=42
        ))
    ])
    svm.fit(X_train, y_train)
    acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"   Accuracy: {acc*100:.2f}%")

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': svm, 'label_encoder': le}, f)
    print(f"Model saved: {MODEL_PATH}")

    y_pred = svm.predict(X_test)
    print(f"\nFinal Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')
    sns.heatmap(cm_pct, annot=True, fmt='.1f',
                xticklabels=le.classes_, yticklabels=le.classes_,
                cmap='Greens', ax=ax, linewidths=0.5,
                linecolor='#0A0F0A')
    ax.set_title('Confusion Matrix — Fixed Model (%)',
                 color='#00FF96', pad=15, fontsize=13)
    ax.set_xlabel('Predicted', color='#88AA88')
    ax.set_ylabel('Actual', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_fixed.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()

    print(f"\nDone! Now run: python realtime_fixed.py")


if __name__ == '__main__':
    train()
