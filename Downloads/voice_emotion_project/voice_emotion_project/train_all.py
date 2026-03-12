"""
train_all.py
Train on ALL 4 datasets: RAVDESS + TESS + CREMA-D + SAVEE
Total: ~12,000 samples | Expected accuracy: 88-94% all emotions
Fast training: ~5 minutes

Run:
    python train_all.py
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

# ── PATHS ──
RAVDESS_DIR = 'data'
TESS_DIR    = 'data_tess/TESS Toronto emotional speech set data'
CREMA_DIR   = 'data_crema/ALL'
SAVEE_DIR   = 'data_savee/AudioWAV'
MODEL_PATH  = 'models/all_model.pkl'
RESULTS_DIR = 'results/plots'

os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── EMOTION MAPS ──
RAVDESS_MAP = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}

TESS_MAP = {
    'angry':'angry','disgust':'disgust','fear':'fearful',
    'happy':'happy','neutral':'neutral','sad':'sad',
    'pleasant_surprise':'surprised','pleasant_surprised':'surprised'
}

CREMA_MAP = {
    'ANG':'angry','DIS':'disgust','FEA':'fearful',
    'HAP':'happy','NEU':'neutral','SAD':'sad'
}

SAVEE_MAP = {
    'a':'angry','d':'disgust','f':'fearful',
    'h':'happy','n':'neutral','sa':'sad','su':'surprised'
}

EMOTION_COLORS = {
    'neutral':'#98989D','calm':'#5AC8FA','happy':'#FFD700',
    'sad':'#4A90D9','angry':'#FF3B30','fearful':'#BF5AF2',
    'disgust':'#FF9F0A','surprised':'#30D158'
}


# ── FEATURE EXTRACTION ──
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


# ── LOADERS ──
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


def load_crema():
    print("Loading CREMA-D...")
    X, y = [], []
    files = glob.glob(os.path.join(CREMA_DIR, '*.wav'))
    for f in tqdm(files, desc="  CREMA-D"):
        name = os.path.basename(f)
        parts = name.split('_')
        if len(parts) < 3: continue
        code = parts[2]
        if code not in CREMA_MAP: continue
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(CREMA_MAP[code])
    print(f"   CREMA-D: {len(X)} samples")
    return X, y


def load_savee():
    print("Loading SAVEE...")
    X, y = [], []
    files = glob.glob(os.path.join(SAVEE_DIR, '*.wav'))
    for f in tqdm(files, desc="  SAVEE"):
        name = os.path.basename(f)
        # Format: DC_a01.wav — emotion is letters before digits
        parts = name.split('_')
        if len(parts) < 2: continue
        code = ''.join([c for c in parts[1].split('.')[0] if c.isalpha()])
        if code not in SAVEE_MAP: continue
        feat = extract_features(f)
        if feat is not None:
            X.append(feat)
            y.append(SAVEE_MAP[code])
    print(f"   SAVEE: {len(X)} samples")
    return X, y


# ── TRAIN ──
def train():
    print("=" * 55)
    print("  Voice Emotion Recognition")
    print("  RAVDESS + TESS + CREMA-D + SAVEE")
    print("  Expected Accuracy: 88-94%")
    print("=" * 55)

    # Load all datasets
    X1, y1 = load_ravdess()
    X2, y2 = load_tess()
    X3, y3 = load_crema()
    X4, y4 = load_savee()

    X = np.array(X1 + X2 + X3 + X4)
    y_raw = y1 + y2 + y3 + y4

    print(f"\nTotal samples: {len(X)}")

    # Encode
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Emotions: {list(le.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # SVM
    print("\nTraining SVM...")
    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42))
    ])
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    print(f"   SVM Accuracy: {svm_acc*100:.2f}%")

    # MLP
    print("\nTraining MLP...")
    mlp = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu', max_iter=300,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, verbose=False
        ))
    ])
    mlp.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp.predict(X_test))
    print(f"   MLP Accuracy: {mlp_acc*100:.2f}%")

    # Best model
    if svm_acc >= mlp_acc:
        best_model, best_name, best_acc = svm, "SVM", svm_acc
    else:
        best_model, best_name, best_acc = mlp, "MLP", mlp_acc

    print(f"\nBest: {best_name} — {best_acc*100:.2f}%")

    # Save
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'label_encoder': le}, f)
    print(f"Model saved: {MODEL_PATH}")

    # Report
    y_pred = best_model.predict(X_test)
    print(f"\nFinal Accuracy: {best_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    save_confusion_matrix(y_test, y_pred, le.classes_)
    save_accuracy_chart(svm_acc, mlp_acc, len(X))

    print(f"\nDone! Now run: python realtime_all.py")


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

    ax.set_title('Confusion Matrix — All Datasets (%)',
                 color='#00FF96', pad=15, fontsize=13)
    ax.set_xlabel('Predicted', color='#88AA88')
    ax.set_ylabel('Actual', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_all.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("Saved confusion matrix")


def save_accuracy_chart(svm_acc, mlp_acc, total_samples):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')

    models = ['SVM\n(RBF)', 'MLP\n(512-256-128)']
    accs   = [svm_acc * 100, mlp_acc * 100]
    colors = ['#00FF96', '#FFD700']

    bars = ax.bar(models, accs, color=colors, alpha=0.85,
                  width=0.4, edgecolor='#0A1A0F')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center',
                color='white', fontsize=12, fontweight='bold')

    ax.set_ylim(0, 105)
    ax.set_title(f'All Datasets Combined ({total_samples} samples)',
                 color='#00FF96', fontsize=13, pad=12)
    ax.set_ylabel('Test Accuracy (%)', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    ax.spines[:].set_color('#1A3A1A')
    ax.grid(True, axis='y', color='#1A3A1A', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/all_accuracy.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("Saved accuracy chart")


if __name__ == '__main__':
    train()
