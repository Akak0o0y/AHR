"""CRNN + CTC for Arabic Handwritten Word Recognition.

Why this replaces segmentation-then-recognition:
  - No explicit character segmentation needed — CTC learns the alignment.
  - Handles ligatures / connected letters naturally.
  - Trains on whole-word images (your 200k corpus), not cropped letters.
  - Variable-length output, variable-width input.

Pipeline:
  word image (H=32, W=variable)
    → CNN backbone (reuses expert_best.keras features)
    → feature map (H'=1, W'=T, C=features)
    → BiLSTM x2 (sequence context, bidirectional for RTL)
    → Dense(num_classes + 1 blank) softmax
    → CTC loss during training / CTC greedy decode at inference
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K


# ── Alphabet ──────────────────────────────────────────────────────────────────
# 28 canonical Arabic letters, single-codepoint each. Uses bare alif (ا) and
# bare ha (ه) — these are the standard NLP-canonical forms that dictionary
# words actually contain. Hamza-variants and other shapes are normalized to
# these canonical forms via NORMALIZE before encoding.
ARABIC_ALPHABET = [
    'ا','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص',
    'ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','ه','و','ي'
]
CHAR_TO_IDX = {c: i for i, c in enumerate(ARABIC_ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ARABIC_ALPHABET)}
NUM_CLASSES = len(ARABIC_ALPHABET)       # 28
BLANK_IDX   = NUM_CLASSES                # CTC blank = 28

# Standard Arabic normalization. Collapses writing-system variants to the
# canonical 28-letter set before encoding. Matches what every serious Arabic
# OCR / NLP system does (ElixirFM, Farasa, CAMeL Tools).
NORMALIZE = str.maketrans({
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا',   # alif-hamza variants → bare alif
    'ى': 'ي',                        # alif maqsura → ya
    'ة': 'ه',                        # taa marbuta → ha
    'ؤ': 'و',                        # waw-hamza → waw
    'ئ': 'ي',                        # ya-hamza → ya
    'ـ': '',                          # tatweel / kashida → drop
    'ء': '',                          # hamza on the line → drop (no glyph)
    # Arabic diacritics (tashkeel) — drop all
    '\u064B': '', '\u064C': '', '\u064D': '',     # fathatan, dammatan, kasratan
    '\u064E': '', '\u064F': '', '\u0650': '',     # fatha, damma, kasra
    '\u0651': '', '\u0652': '', '\u0653': '',     # shadda, sukun, madda
    '\u0654': '', '\u0655': '', '\u0670': '',     # hamza above/below, dagger alif
})

IMG_H = 32            # fixed height
IMG_W = 128           # target width for training (pad/resize)
DOWNSAMPLE = 4        # CNN stride → time steps T = IMG_W // DOWNSAMPLE


# ── Label encoding ────────────────────────────────────────────────────────────
def encode_label(word: str) -> np.ndarray:
    """Arabic string → list of class indices.

    Applies canonical Arabic normalization first (alif-hamza variants → ا,
    taa marbuta → ه, drop diacritics/tatweel), then maps to alphabet indices.
    Silently drops any characters still outside the alphabet after normalization.
    """
    normalized = word.translate(NORMALIZE)
    return np.array([CHAR_TO_IDX[c] for c in normalized if c in CHAR_TO_IDX],
                    dtype=np.int32)


def decode_prediction(logits: np.ndarray) -> str:
    """CTC greedy decode: collapse repeats, drop blanks."""
    best = np.argmax(logits, axis=-1)        # (T,)
    out, prev = [], -1
    for idx in best:
        if idx != prev and idx != BLANK_IDX:
            out.append(IDX_TO_CHAR[idx])
        prev = idx
    return ''.join(out)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_crnn(img_h=IMG_H, img_w=IMG_W, num_classes=NUM_CLASSES,
               backbone_weights: str | None = None) -> tf.keras.Model:
    """CNN → BiLSTM → softmax.

    Args:
      backbone_weights: path to expert_best.keras to warm-start the CNN.
                        None = train from scratch.
    """
    inp = layers.Input(shape=(img_h, img_w, 1), name='image')

    # CNN backbone — mirrors expert model's early layers for weight transfer.
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)                    # H/2, W/2

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)                    # H/4, W/4

    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)                    # H/8, W/4 — keep W

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)                    # H/16, W/4

    x = layers.Conv2D(256, (2, 2), padding='valid', activation='relu')(x)
    # Collapse height → sequence of W/4 feature vectors.
    x = layers.Reshape((-1, 256))(x)                      # (T, 256)

    # BiLSTM for temporal context (bidirectional is important for Arabic
    # because diacritics / dots above-below influence letter identity).
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    # +1 for CTC blank
    logits = layers.Dense(num_classes + 1, activation='softmax', name='logits')(x)

    model = models.Model(inp, logits, name='CRNN_Arabic')

    if backbone_weights and os.path.exists(backbone_weights):
        expert = tf.keras.models.load_model(backbone_weights)
        for src, dst in zip(expert.layers, model.layers):
            if isinstance(src, (layers.Conv2D, layers.BatchNormalization)) and \
               src.get_config().get('filters') == dst.get_config().get('filters'):
                try:
                    dst.set_weights(src.get_weights())
                except ValueError:
                    pass
        print(f"Warm-started CNN from {backbone_weights}")

    return model


# ── CTC loss wrapper ──────────────────────────────────────────────────────────
def ctc_loss_fn(y_true, y_pred):
    """y_true shape: (B, L) padded with -1.  y_pred shape: (B, T, C+1)."""
    batch = tf.shape(y_pred)[0]
    input_len  = tf.fill([batch, 1], tf.shape(y_pred)[1])
    label_len  = tf.math.count_nonzero(y_true + 1, axis=-1, keepdims=True,
                                       dtype=tf.int32)
    return K.ctc_batch_cost(y_true, y_pred, input_len, label_len)


# ── Training entry point ──────────────────────────────────────────────────────
def train(X_words, y_labels, val_split=0.1, epochs=50, batch_size=32,
          backbone_weights='expert_best.keras',
          out_path='crnn_arabic.keras'):
    """
    Args:
      X_words: (N, 32, 128, 1) float32 in [0,1] — resized word images.
      y_labels: list of np.ndarray, each a 1-D array of class indices
                (from encode_label). Will be right-padded with -1.
    """
    max_len = max(len(y) for y in y_labels)
    y_padded = np.full((len(y_labels), max_len), -1, dtype=np.int32)
    for i, y in enumerate(y_labels):
        y_padded[i, :len(y)] = y

    model = build_crnn(backbone_weights=backbone_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=ctc_loss_fn)
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
        tf.keras.callbacks.ModelCheckpoint(out_path, save_best_only=True,
                                           monitor='val_loss'),
    ]
    hist = model.fit(X_words, y_padded, validation_split=val_split,
                     epochs=epochs, batch_size=batch_size, callbacks=cbs)
    print(f"Saved → {out_path}")
    return model, hist


# ── Inference ─────────────────────────────────────────────────────────────────
def predict_word(model, img: np.ndarray) -> str:
    """img: grayscale (H,W) or (H,W,1), any size — resized to (32,128)."""
    import cv2
    if img.ndim == 3:
        img = img[..., 0]
    resized = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = resized.astype('float32') / 255.0
    x = x.reshape(1, IMG_H, IMG_W, 1)
    logits = model.predict(x, verbose=0)[0]     # (T, C+1)
    return decode_prediction(logits)


# ── Guidance: still want better segmentation too? ─────────────────────────────
# If you must keep a segmentation pathway (e.g. for data weakly-labeled at the
# character level), these help on connected Arabic:
#   1. Projection-profile segmentation on the baseline strip: compute vertical
#      pixel-sum profile, find minima below the baseline — those are the
#      inter-letter "necks" of the kashida connector.
#   2. Stroke-width transform (SWT): split on abrupt stroke-width changes.
#   3. Dots-aware merging: detect small components whose bbox sits above/below
#      a larger base component and merge them before classification.
# But none of these beat CTC on real data. Use them only as a fallback.

if __name__ == '__main__':
    # Smoke test — build the model, show the output shape.
    m = build_crnn()
    m.summary()
    dummy = np.zeros((2, IMG_H, IMG_W, 1), dtype='float32')
    print("output shape:", m.predict(dummy, verbose=0).shape)
