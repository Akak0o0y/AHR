"""Aggressive preprocessing for real handwritten word images.

Real photographed/scanned Arabic handwriting differs from the synth corpus
along several axes:
  - low contrast (light pen on tinted paper vs. pure black-on-white)
  - paper texture (vs. flat white background)
  - aspect ratios that put the word in a small portion of the frame
  - varied orientation (sometimes the word reads vertically in the image)

This script tries a fixed bag of preprocessing variants and reports the
prediction for each so we can spot whether ANY preprocessing helps.

Usage:
    python src/predict_real.py data/raw/real_test/img.jpg
    python src/predict_real.py data/raw/real_test/  --label "مدينة"
"""
from __future__ import annotations
import argparse, sys, io, os
import numpy as np
import cv2
import tensorflow as tf

from solution import ctc_loss_fn, decode_prediction
from predict import _preprocess
import paths


def _otsu(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _stretch(g):
    lo, hi = np.percentile(g, 5), np.percentile(g, 95)
    return np.clip((g.astype(np.float32) - lo) * 255 / max(1, hi - lo),
                   0, 255).astype(np.uint8)


def _tight_crop(g, ink_thr=200):
    mask = g < ink_thr
    if not mask.any():
        return g
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    pad = max(2, max(y1 - y0, x1 - x0) // 12)
    y0 = max(0, y0 - pad); y1 = min(g.shape[0], y1 + pad)
    x0 = max(0, x0 - pad); x1 = min(g.shape[1], x1 + pad)
    return g[y0:y1, x0:x1]


def variants(path: str) -> dict[str, np.ndarray]:
    """Generate a bag of preprocessing variants for one image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    out: dict[str, np.ndarray] = {}
    # Try as-is + 4 rotations × 3 enhancements = 12 variants
    rotations = {
        '0':   img,
        '90':  cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        '180': cv2.rotate(img, cv2.ROTATE_180),
        '270': cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    for r, rimg in rotations.items():
        out[f'r{r}_raw']     = _tight_crop(rimg)
        out[f'r{r}_stretch'] = _tight_crop(_stretch(rimg))
        out[f'r{r}_otsu']    = _tight_crop(_otsu(rimg))
    return out


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--model', default='models/crnn_arabic_5k.keras')
    p.add_argument('--label', default=None,
                   help='Ground-truth word; if set, prints which variants '
                        'matched (✓) vs. missed (✗).')
    args = p.parse_args()

    print(f"Loading {args.model}")
    model = tf.keras.models.load_model(args.model,
        custom_objects={'ctc_loss_fn': ctc_loss_fn})

    files = ([os.path.join(args.path, f) for f in sorted(os.listdir(args.path))
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
             if os.path.isdir(args.path) else [args.path])

    for fp in files:
        print(f"\n=== {os.path.basename(fp)} ===")
        vs = variants(fp)
        # Stack all variants through the model in one batch.
        batch = np.concatenate([_preprocess(g) for g in vs.values()], axis=0)
        preds = model.predict(batch, verbose=0)
        for (name, _), p in zip(vs.items(), preds):
            d = decode_prediction(p)
            mark = ''
            if args.label is not None:
                mark = ' ✓' if d == args.label else ' ✗'
            print(f"  {name:14s} → {d!r}{mark}")


if __name__ == '__main__':
    main()
