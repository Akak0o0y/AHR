"""Train the CRNN from solution.py on a synth corpus.

Pipeline:
  synth corpus npz (variable-width images, string labels)
    → resize each image to (32, 128) — IMG_H × IMG_W from solution.py
    → encode labels as padded int sequences
    → feed into CRNN + CTC
    → save crnn_arabic.keras

Run:
    python scripts/train.py --corpus data/synth/synth_corpus_dict.npz --epochs 25
    python scripts/train.py --smoke                                     # 1-epoch sanity check

History note: the original main() used EarlyStopping with a validation split.
On small synth corpora the val_loss climbs from epoch 2 onward (small val set
drifts from train distribution), and EarlyStopping's restore_best_weights then
reverts the model to its near-random epoch-1 weights — CTC collapses to
all-blanks. The current main() drops the val split and early stop entirely;
ReduceLROnPlateau on the train loss is the only learning-rate dynamic we keep.
"""

from __future__ import annotations
import argparse
import sys
import io
import numpy as np
import cv2
import tensorflow as tf

from solution import (
    build_crnn, ctc_loss_fn, encode_label, decode_prediction,
    IMG_H, IMG_W,
)
import paths


# ── Data prep ────────────────────────────────────────────────────────────────
def resize_keep_aspect(img: np.ndarray, target_h: int = IMG_H,
                       target_w: int = IMG_W) -> np.ndarray:
    """Resize to target height, keep aspect, pad or crop width to target_w.

    Pad value is 255 (white, matches compose_word output background).
    """
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    if new_w >= target_w:
        return resized[:, :target_w]                # center-crop would also work
    out = np.full((target_h, target_w), 255, dtype=np.uint8)
    out[:, :new_w] = resized
    return out


def load_corpus(path: str):
    """Return (X, y_padded, max_label_len)."""
    d = np.load(path, allow_pickle=True)
    images, labels = d['images'], d['labels']
    print(f"Loaded {len(images)} samples from {path}")

    X = np.zeros((len(images), IMG_H, IMG_W, 1), dtype='float32')
    y_list: list[np.ndarray] = []
    for i, (im, lbl) in enumerate(zip(images, labels)):
        X[i, :, :, 0] = resize_keep_aspect(im).astype('float32') / 255.0
        y_list.append(encode_label(str(lbl)))

    max_len = max(len(y) for y in y_list)
    y = np.full((len(y_list), max_len), -1, dtype=np.int32)
    for i, yi in enumerate(y_list):
        y[i, :len(yi)] = yi

    # Synth images are already RTL from word_composer (first letter on right),
    # but for CTC we want image order to match label order (first letter first).
    # Horizontally flip so leftmost pixel-column = first char in label.
    # At inference on real RTL handwriting, apply the same flip.
    X = X[:, :, ::-1, :]

    print(f"  X: {X.shape}  y: {y.shape}  max_label_len: {max_len}")
    return X, y, max_len


# ── Evaluate a few predictions for a sanity peek ──────────────────────────────
def peek(model, X, y, n: int = 8):
    """Greedy-decode a few predictions; compare to true labels."""
    from solution import IDX_TO_CHAR
    idx = np.random.choice(len(X), size=min(n, len(X)), replace=False)
    preds = model.predict(X[idx], verbose=0)
    for i, p in zip(idx, preds):
        true_ids = y[i][y[i] != -1]
        true = ''.join(IDX_TO_CHAR[int(t)] for t in true_ids)
        pred = decode_prediction(p)
        marker = '✓' if pred == true else '✗'
        print(f"  {marker}  true={true!r:15s}  pred={pred!r}")


# ── Entry ─────────────────────────────────────────────────────────────────────
def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default=paths.SYNTH_CORPUS_ENRICHED)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--backbone', default=paths.EXPERT_MODEL,
                   help='Warm-start CNN weights; empty to train from scratch.')
    p.add_argument('--out', default=paths.CRNN_MODEL)
    p.add_argument('--smoke', action='store_true',
                   help='1-epoch sanity check, no callbacks.')
    args = p.parse_args()

    X, y, _ = load_corpus(args.corpus)

    model = build_crnn(backbone_weights=args.backbone or None)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=ctc_loss_fn)

    if args.smoke:
        print("\n=== SMOKE TRAIN (1 epoch, no callbacks) ===")
        model.fit(X, y, batch_size=min(args.batch_size, len(X)),
                  epochs=1, verbose=1)
        print("\nPost-smoke peek (expect mostly gibberish — only 1 epoch):")
        peek(model, X, y)
        return

    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=6,
                                             factor=0.5, min_lr=1e-5),
    ]
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size,
              callbacks=cbs, verbose=2)
    model.save(paths.ensure_dir(args.out))
    print(f"\nSaved → {args.out}")
    print("\nFinal peek (greedy decode on training data):")
    peek(model, X, y, n=12)


if __name__ == '__main__':
    main()
