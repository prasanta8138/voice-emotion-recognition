"""
train_model_svm.py
Train SVM + MLP on RAVDESS — works much better on small datasets.
Expected accuracy: 70-85%

Run:
    python train_model_svm.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import glob
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import pickle

DATA_DIR    = 'data'
MODEL_PATH  = 'models/svm_model.pkl'
RESULTS_DIR = 'results/plots'

os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry',   '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

EMOTION_COLORS = {
    'neutral':'#98989D', 'calm':'#5AC8FA', 'happy':'#FFD700',
    'sad':'#4A90D9', 'angry':'#FF3B30', 'fearful':'#BF5AF2',
    'disgust':'#FF9F0A', 'surprised':'#30D158'
}


def extract_features_flat(file_path):
    """
    Extract flat feature vector for SVM:
    - MFCC mean + std (40x2 = 80)
    - Chroma mean + std (12x2 = 24)
    - Mel mean + std (128x2 = 256)
    - ZCR mean + std (2)
    - RMS mean + std (2)
    - Spectral centroid mean + std (2)
    - Spectral rolloff mean + std (2)
    Total: 368 features
    """
    try:
        y, sr = librosa.load(file_path, duration=3, res_type='kaiser_fast')

        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # Chroma
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # RMS energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(cent))
        features.append(np.std(cent))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        return np.array(features)

    except Exception as e:
        return None


def load_dataset():
    print("📂 Loading RAVDESS dataset...")
    wav_files = glob.glob(os.path.join(DATA_DIR, 'Actor_*', '*.wav'))

    if len(wav_files) == 0:
        print("❌ No audio files found!")
        return None, None, None

    print(f"   Found {len(wav_files)} audio files")

    X, y = [], []
    for f in tqdm(wav_files, desc="Extracting features"):
        parts = os.path.basename(f).split('-')
        if len(parts) < 3:
            continue
        code = parts[2]
        if code not in EMOTION_MAP:
            continue
        feat = extract_features_flat(f)
        if feat is not None:
            X.append(feat)
            y.append(EMOTION_MAP[code])

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"\n✅ Loaded {len(X)} samples, {len(le.classes_)} emotions")
    return X, y_enc, le


def train():
    print("=" * 55)
    print("  🎙️  Voice Emotion Recognition — SVM + MLP")
    print("  Dataset: RAVDESS | Features: MFCC+Chroma+Mel")
    print("=" * 55)

    X, y, le = load_dataset()
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Model 1: SVM ──
    print("\n🔧 Training SVM...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm_pipeline.predict(X_test))
    print(f"   SVM Accuracy: {svm_acc*100:.2f}%")

    # ── Model 2: MLP ──
    print("\n🔧 Training MLP Neural Network...")
    mlp_pipeline = Pipeline([
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
    mlp_pipeline.fit(X_train, y_train)
    mlp_acc = accuracy_score(y_test, mlp_pipeline.predict(X_test))
    print(f"   MLP Accuracy: {mlp_acc*100:.2f}%")

    # Pick best model
    if svm_acc >= mlp_acc:
        best_model = svm_pipeline
        best_name = "SVM"
        best_acc = svm_acc
    else:
        best_model = mlp_pipeline
        best_name = "MLP"
        best_acc = mlp_acc

    print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

    # Save best model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({'model': best_model, 'label_encoder': le}, f)
    print(f"✅ Model saved to: {MODEL_PATH}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    print(f"\n✅ Test Accuracy: {best_acc*100:.2f}%")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save plots
    save_confusion_matrix(y_test, y_pred, le.classes_)
    save_comparison_plot(svm_acc, mlp_acc)

    print(f"\n✅ Done! Run: python realtime_svm.py")


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

    ax.set_title('Confusion Matrix (%)', color='#00FF96', pad=15, fontsize=13)
    ax.set_xlabel('Predicted', color='#88AA88')
    ax.set_ylabel('Actual', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix_svm.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved confusion matrix")


def save_comparison_plot(svm_acc, mlp_acc):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')

    models = ['SVM\n(RBF Kernel)', 'MLP\n(512-256-128)']
    accs = [svm_acc * 100, mlp_acc * 100]
    colors = ['#00FF96', '#FFD700']

    bars = ax.bar(models, accs, color=colors, alpha=0.85, width=0.4,
                  edgecolor='#0A1A0F')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', color='white',
                fontsize=12, fontweight='bold')

    ax.set_ylim(0, 100)
    ax.set_title('Model Comparison on RAVDESS',
                 color='#00FF96', fontsize=13, pad=12)
    ax.set_ylabel('Test Accuracy (%)', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    ax.spines[:].set_color('#1A3A1A')
    ax.grid(True, axis='y', color='#1A3A1A', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/svm_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved model comparison")


if __name__ == '__main__':
    train()
