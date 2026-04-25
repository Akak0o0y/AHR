"""Run the trained CRNN on an arbitrary image of a handwritten Arabic word.

The model was trained on synth data with these conventions:
  - Grayscale, white background, dark ink.
  - 32 px tall × 128 px wide; aspect preserved by resizing to height 32 then
    right-padding with white to 128.
  - Global horizontal flip (image columns then run LEFT→RIGHT in label order),
    so CTC decodes a word in label-character order.

This script applies the same preprocessing to ANY input image (any size,
any format), runs the model, and prints the greedy CTC decoding.

Usage:
    python scripts/predict.py path/to/word.png
    python scripts/predict.py path/to/word.png --no-invert     # don't auto-invert
    python scripts/predict.py path/to/dir/                     # batch on a folder
"""
from __future__ import annotations
import argparse, sys, io, os
import numpy as np
import cv2
import tensorflow as tf

from solution import ctc_loss_fn, decode_prediction, IMG_H, IMG_W
import paths


def _load_image_gray(path: str, auto_invert: bool = True) -> np.ndarray:
    """Read any image format → uint8 grayscale; auto-invert if dark bg."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:                                       # RGBA
            alpha = img[..., 3:4] / 255.0
            rgb = img[..., :3].astype(np.float32)
            white = np.full_like(rgb, 255)
            img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if auto_invert and int(np.median(img)) < 128:
        img = 255 - img
    return img


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Match the training pipeline: resize to (IMG_H, IMG_W), normalize, flip."""
    h, w = img.shape[:2]
    scale = IMG_H / h
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(img, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    if new_w >= IMG_W:
        canvas = resized[:, :IMG_W]
    else:
        canvas = np.full((IMG_H, IMG_W), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized
    x = canvas.astype('float32') / 255.0
    x = x[:, ::-1]                                                  # global RTL flip
    return x[None, :, :, None]                                      # (1, H, W, 1)


def predict_one(model, path: str, auto_invert: bool = True,
                save_preview: bool = False) -> str:
    img = _load_image_gray(path, auto_invert=auto_invert)
    x = _preprocess(img)
    if save_preview:
        prev = (x[0, :, :, 0] * 255).astype('uint8')[:, ::-1]
        out_dir = paths.ensure_dir(f"{paths.OUTPUTS_DIR}/predict_preview/x.png")
        out_png = os.path.join(os.path.dirname(out_dir),
                               os.path.splitext(os.path.basename(path))[0] + '_in.png')
        cv2.imwrite(out_png, prev)
    logits = model.predict(x, verbose=0)[0]
    return decode_prediction(logits)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('path', help='Image file OR directory of images.')
    p.add_argument('--model', default=paths.CRNN_MODEL)
    p.add_argument('--no-invert', dest='auto_invert', action='store_false',
                   help='Disable auto-invert (use if your image already has '
                        'white bg + dark ink and the heuristic gets it wrong).')
    p.add_argument('--save-preview', action='store_true',
                   help='Save the preprocessed (32×128, flipped) image we '
                        'fed the model to outputs/predict_preview/.')
    args = p.parse_args()

    print(f"Loading {args.model}")
    model = tf.keras.models.load_model(args.model,
        custom_objects={'ctc_loss_fn': ctc_loss_fn})

    if os.path.isdir(args.path):
        exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
        files = sorted(f for f in os.listdir(args.path) if f.lower().endswith(exts))
        if not files:
            print(f"No image files found in {args.path}")
            return
        for fn in files:
            full = os.path.join(args.path, fn)
            pred = predict_one(model, full, args.auto_invert, args.save_preview)
            print(f"  {fn:40s} → pred = {pred!r}")
    else:
        pred = predict_one(model, args.path, args.auto_invert, args.save_preview)
        print(f"\nPrediction: {pred!r}")


if __name__ == '__main__':
    main()
