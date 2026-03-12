"""
train_combined.py
Train SVM + MLP on RAVDESS + TESS combined dataset.
Expected accuracy: 85-93%

Run:
    python train_combined.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAVDESS_DIR = 'data'
TESS_DIR    = 'data_tess/TESS Toronto emotional speech set data'
MODEL_PATH  = 'models/combined_model.pkl'
RESULTS_DIR = 'results/plots'

os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# RAVDESS emotion codes
RAVDESS_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

# TESS folder name to emotion
TESS_MAP = {
    'angry': 'angry', 'disgust': 'disgust', 'fear': 'fearful',
    'happy': 'happy', 'neutral': 'neutral', 'sad': 'sad',
    'pleasant_surprise': 'surprised', 'pleasant_surprised': 'surprised'
}

EMOTION_COLORS = {
    'neutral':'#98989D', 'calm':'#5AC8FA', 'happy':'#FFD700',
    'sad':'#4A90D9', 'angry':'#FF3B30', 'fearful':'#BF5AF2',
    'disgust':'#FF9F0A', 'surprised':'#30D158'
}


# ─────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────
def extract_features(file_path, sr_target=22050):
    """Extract 368-dimensional feature vector."""
    try:
        y, sr = librosa.load(file_path, duration=3, res_type='kaiser_fast')
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
            sr = sr_target

        features = []

        # MFCC (80 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Chroma (24 features)
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Mel spectrogram (256 features)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        # ZCR (2)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # RMS (2)
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # Spectral centroid (2)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(cent))
        features.append(np.std(cent))

        # Spectral rolloff (2)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        return np.array(features)

    except Exception as e:
        return None


# ─────────────────────────────────────────
# LOAD RAVDESS
# ─────────────────────────────────────────
def load_ravdess():
    print("📂 Loading RAVDESS...")
    files = glob.glob(os.path.join(RAVDESS_DIR, 'Actor_*', '*.wav'))
    X, y = [], []

    for f in tqdm(files, desc="  RAVDESS"):
        parts = os.path.basename(f).split('-')
        if len(parts) < 3:
            continue
        code = parts[2]
        if code not in RAVDESS_MAP:
            continue
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(RAVDESS_MAP[code])

    print(f"   ✅ RAVDESS: {len(X)} samples")
    return X, y


# ─────────────────────────────────────────
# LOAD TESS
# ─────────────────────────────────────────
def load_tess():
    print("📂 Loading TESS...")
    X, y = [], []

    folders = glob.glob(os.path.join(TESS_DIR, '*'))
    for folder in folders:
        folder_name = os.path.basename(folder).lower()

        # Extract emotion from folder name
        emotion = None
        for key, val in TESS_MAP.items():
            if key in folder_name:
                emotion = val
                break

        if emotion is None:
            continue

        wav_files = glob.glob(os.path.join(folder, '*.wav'))
        for f in tqdm(wav_files, desc=f"  {emotion}", leave=False):
            feat = extract_features(f)
            if feat is not None:
                X.append(feat)
                y.append(emotion)

    print(f"   ✅ TESS: {len(X)} samples")
    return X, y


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train():
    print("=" * 55)
    print("  🎙️  Voice Emotion Recognition")
    print("  Dataset: RAVDESS + TESS Combined")
    print("  Expected Accuracy: 85-93%")
    print("=" * 55)

    # Load both datasets
    X_r, y_r = load_ravdess()
    X_t, y_t = load_tess()

    # Combine
    X = np.array(X_r + X_t)
    y_raw = y_r + y_t

    print(f"\n📊 Combined dataset: {len(X)} total samples")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"   Emotions: {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"   Train: {len(X_train)} | Test: {len(X_test)}")

    # ── SVM ──
    print("\n🔧 Training SVM (RBF kernel)...")
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42))
    ])
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"   SVM Accuracy: {svm_acc*100:.2f}%")

    # ── MLP ──
    print("\n🔧 Training MLP Neural Network...")
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ))
    ])
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"   MLP Accuracy: {mlp_acc*100:.2f}%")

    # Pick best
    if svm_acc >= mlp_acc:
        best_model, best_name, best_acc = svm, "SVM", svm_acc
    else:
        best_model, best_name, best_acc = mlp, "MLP", mlp_acc

    print(f"\n🏆 Best Model: {best_name} — {best_acc*100:.2f}%")

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'label_encoder': le}, f)
    print(f"✅ Model saved to: {MODEL_PATH}")

    # Report
    y_pred = best_model.predict(X_test)
    print(f"\n✅ Final Test Accuracy: {best_acc*100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Plots
    save_confusion_matrix(y_test, y_pred, le.classes_)
    save_accuracy_chart(svm_acc, mlp_acc)

    print(f"\n✅ All done!")
    print(f"   Now run: python realtime_combined.py")


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')

    sns.heatmap(cm_pct, annot=True, fmt='.1f',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Greens', ax=ax, linewidths=0.5,
                linecolor='#0A0F0A', cbar_kws={'label': 'Accuracy %'})

    ax.set_title('Confusion Matrix — RAVDESS + TESS (%)',
                 color='#00FF96', pad=15, fontsize=13)
    ax.set_xlabel('Predicted', color='#88AA88')
    ax.set_ylabel('Actual', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_combined.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved confusion matrix")


def save_accuracy_chart(svm_acc, mlp_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')

    models = ['SVM\n(RBF Kernel)', 'MLP\n(512-256-128)']
    accs = [svm_acc * 100, mlp_acc * 100]
    colors = ['#00FF96', '#FFD700']

    bars = ax.bar(models, accs, color=colors, alpha=0.85,
                  width=0.4, edgecolor='#0A1A0F')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center',
                color='white', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_title('RAVDESS + TESS — Model Accuracy',
                 color='#00FF96', fontsize=13, pad=12)
    ax.set_ylabel('Test Accuracy (%)', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    ax.spines[:].set_color('#1A3A1A')
    ax.grid(True, axis='y', color='#1A3A1A', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/combined_accuracy.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved accuracy chart")


if __name__ == '__main__':
    train()
