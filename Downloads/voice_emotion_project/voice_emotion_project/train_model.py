"""
train_model.py
Train CNN + BiLSTM on RAVDESS with Data Augmentation.
Expected accuracy: 75-88%

Run:
    python train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Reshape, LSTM, Dense, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix
from utils.preprocess import load_ravdess_dataset, get_train_test_split, EMOTIONS

DATA_DIR    = 'data'
MODEL_PATH  = 'models/emotion_model.h5'
RESULTS_DIR = 'results/plots'
EPOCHS      = 100
BATCH_SIZE  = 16
NUM_CLASSES = 8

os.makedirs('models', exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def augment_features(X, y):
    print(f"\n🔀 Augmenting training data...")
    X_aug = [X]
    y_aug = [y]

    # 1. Gaussian noise (soft)
    X_aug.append(X + np.random.normal(0, 0.015, X.shape))
    y_aug.append(y)

    # 2. Gaussian noise (stronger)
    X_aug.append(X + np.random.normal(0, 0.03, X.shape))
    y_aug.append(y)

    # 3. Time masking
    X_time = X.copy()
    for j in range(len(X_time)):
        t = np.random.randint(10, 40)
        t0 = np.random.randint(0, X.shape[2] - t)
        X_time[j, :, t0:t0+t, :] = 0
    X_aug.append(X_time)
    y_aug.append(y)

    # 4. Frequency masking
    X_freq = X.copy()
    for j in range(len(X_freq)):
        f = np.random.randint(5, 30)
        f0 = np.random.randint(0, X.shape[1] - f)
        X_freq[j, f0:f0+f, :, :] = 0
    X_aug.append(X_freq)
    y_aug.append(y)

    # 5. Combined masking
    X_both = X.copy()
    for j in range(len(X_both)):
        t = np.random.randint(5, 25)
        t0 = np.random.randint(0, X.shape[2] - t)
        f = np.random.randint(5, 20)
        f0 = np.random.randint(0, X.shape[1] - f)
        X_both[j, :, t0:t0+t, :] = 0
        X_both[j, f0:f0+f, :, :] = 0
    X_aug.append(X_both)
    y_aug.append(y)

    X_out = np.vstack(X_aug)
    y_out = np.vstack(y_aug)
    idx = np.random.permutation(len(X_out))
    print(f"   Original: {len(X)} → Augmented: {len(X_out)} samples ✅")
    return X_out[idx], y_out[idx]


def build_model(input_shape, num_classes):
    reg = tf.keras.regularizers.l2(0.0005)
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (3,3), activation='relu', padding='same', kernel_regularizer=reg)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,4))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,4))(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)

    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.4, recurrent_dropout=0.3))(x)
    x = Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.3))(x)

    x = Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)


def train():
    print("=" * 55)
    print("  🎙️  Voice Emotion Recognition — CNN + BiLSTM")
    print("  Dataset: RAVDESS | Data Augmentation: ON")
    print("=" * 55)

    X, y, le = load_ravdess_dataset(DATA_DIR)
    if X is None:
        return

    X_train, X_val, X_test, y_train, y_val, y_test = get_train_test_split(X, y)
    X_train, y_train = augment_features(X_train, y_train)

    input_shape = X_train.shape[1:]
    print(f"\n🧠 Building CNN+BiLSTM model...")
    model = build_model(input_shape, NUM_CLASSES)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-7, verbose=1)
    ]

    print(f"\n🚀 Training for up to {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\n📊 Evaluating on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {accuracy * 100:.2f}%")
    print(f"   Test Loss:     {loss:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=le.classes_))

    save_training_plots(history)
    save_confusion_matrix(y_true_classes, y_pred_classes, le.classes_)

    print(f"\n✅ Model saved to: {MODEL_PATH}")
    print(f"   Plots saved to: {RESULTS_DIR}/")


def save_training_plots(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0A0F0A')
    for ax in axes:
        ax.set_facecolor('#0D150D')
        ax.tick_params(colors='#88AA88')
        ax.spines[:].set_color('#1A3A1A')

    axes[0].plot(history.history['accuracy'], color='#00FF96', linewidth=2, label='Train')
    axes[0].plot(history.history['val_accuracy'], color='#FFD700', linewidth=2, label='Validation')
    axes[0].set_title('Model Accuracy', color='#E0FFE0', pad=10)
    axes[0].set_xlabel('Epoch', color='#88AA88')
    axes[0].set_ylabel('Accuracy', color='#88AA88')
    axes[0].legend(facecolor='#0D150D', labelcolor='white')
    axes[0].grid(True, color='#1A3A1A', alpha=0.5)

    axes[1].plot(history.history['loss'], color='#FF3B30', linewidth=2, label='Train')
    axes[1].plot(history.history['val_loss'], color='#FF9F0A', linewidth=2, label='Validation')
    axes[1].set_title('Model Loss', color='#E0FFE0', pad=10)
    axes[1].set_xlabel('Epoch', color='#88AA88')
    axes[1].set_ylabel('Loss', color='#88AA88')
    axes[1].legend(facecolor='#0D150D', labelcolor='white')
    axes[1].grid(True, color='#1A3A1A', alpha=0.5)

    plt.suptitle('CNN+BiLSTM — RAVDESS Speech Emotion Recognition', color='#00FF96', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/training_curves.png', dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved training curves")


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0A0F0A')
    ax.set_facecolor('#0D150D')

    sns.heatmap(cm_pct, annot=True, fmt='.1f',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Greens', ax=ax, linewidths=0.5, linecolor='#0A0F0A',
                cbar_kws={'label': 'Accuracy %'})

    ax.set_title('Confusion Matrix (%)', color='#00FF96', pad=15, fontsize=13)
    ax.set_xlabel('Predicted', color='#88AA88')
    ax.set_ylabel('Actual', color='#88AA88')
    ax.tick_params(colors='#88AA88')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0A0F0A')
    plt.close()
    print("   → Saved confusion matrix")


if __name__ == '__main__':
    train()
