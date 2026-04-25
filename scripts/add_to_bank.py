"""Append handwritten letter samples to the glyph bank npz.

Two modes:

1. Single file:
    python scripts/add_to_bank.py path/to/img.png ب Initial

2. Directory (batch — file names encode (char, form), e.g. `ب_Initial_01.png`):
    python scripts/add_to_bank.py path/to/dir/

Pre-processing matches the rest of the bank:
  - Convert to single-channel grayscale.
  - Auto-invert if background looks dark (we want WHITE bg, DARK ink).
  - Tight-crop to ink bounding box.
  - Pad to a square, resize to 32×32 (cv2.INTER_AREA for downscale).

The original npz is backed up to <name>.bak.npz before the first append in a run.
"""
from __future__ import annotations
import argparse, sys, io, os, re, shutil
import numpy as np
import cv2
import paths


TARGET_HW = 32                         # bank glyphs are 32×32
PAD_FRAC = 0.10                        # padding around ink before resize


def _load_image_gray(path: str) -> np.ndarray:
    """Read any image format → uint8 grayscale; auto-invert if dark bg."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:                          # RGBA → composite over white
            alpha = img[..., 3:4] / 255.0
            rgb = img[..., :3].astype(np.float32)
            white = np.full_like(rgb, 255)
            img = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # Bank convention: white background, dark ink. If median pixel < 128 the
    # background is dark, so invert.
    if int(np.median(img)) < 128:
        img = 255 - img
    return img


def _normalize_glyph(img: np.ndarray, target_hw: int = TARGET_HW) -> np.ndarray:
    """Tight-crop ink, pad to a square, resize to (target_hw, target_hw)."""
    mask = img < 200                                   # ink = anything not near-white
    if not mask.any():
        return cv2.resize(img, (target_hw, target_hw), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = img[y0:y1, x0:x1]

    h, w = cropped.shape
    side = int(round(max(h, w) * (1 + PAD_FRAC * 2)))
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = cropped
    return cv2.resize(canvas, (target_hw, target_hw),
                      interpolation=cv2.INTER_AREA)


# Filename pattern for batch mode: <char>_<form>_<anything>.<ext>
# Forms accepted: Isolated, Initial, Medial, Final
_FORM_RE = re.compile(r'^(?P<char>.+?)_(?P<form>Isolated|Initial|Medial|Final)'
                      r'(?:_[^.]*)?\.(?:png|jpg|jpeg|bmp|tif|tiff|webp)$',
                      re.IGNORECASE)


def _parse_batch_filename(name: str) -> tuple[str, str] | None:
    m = _FORM_RE.match(name)
    if not m:
        return None
    form = m.group('form').capitalize()
    return m.group('char'), form


def _backup_once(npz_path: str) -> str:
    bak = npz_path.replace('.npz', '.bak.npz')
    if not os.path.exists(bak):
        shutil.copy2(npz_path, bak)
        print(f"  backup → {bak}")
    return bak


def append_to_bank(items: list[tuple[np.ndarray, str, str]],
                   npz_path: str = paths.GLYPH_BANK_NPZ) -> None:
    """Append (img, char, form) triples to the bank npz.

    All images must already be 32×32 uint8 grayscale.
    """
    if not items:
        print("Nothing to add.")
        return
    _backup_once(npz_path)
    d = np.load(npz_path, allow_pickle=True)
    images, chars, forms = d['images'], d['char_labels'], d['form_labels']

    # Existing images may be (N, 32, 32, 1) float32 or (N, 32, 32) uint8.
    # Normalize to (N, 32, 32) uint8 so we can stack.
    if images.ndim == 4:
        images = images[..., 0]
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)

    new_imgs = np.stack([it[0] for it in items])
    new_chars = np.array([it[1] for it in items])
    new_forms = np.array([it[2] for it in items])

    out_imgs = np.concatenate([images, new_imgs], axis=0)
    out_chars = np.concatenate([chars, new_chars])
    out_forms = np.concatenate([forms, new_forms])

    np.savez_compressed(npz_path,
                        images=out_imgs,
                        char_labels=out_chars,
                        form_labels=out_forms)
    print(f"\nAppended {len(items)} sample(s) → {npz_path}")
    print(f"  bank size: {len(images)} → {len(out_imgs)}")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('path', help='Image file OR directory (batch mode).')
    p.add_argument('char', nargs='?', default=None,
                   help='Letter (single-file mode). Omit in batch mode.')
    p.add_argument('form', nargs='?', default=None,
                   choices=['Isolated', 'Initial', 'Medial', 'Final', None],
                   help='Form (single-file mode). Omit in batch mode.')
    args = p.parse_args()

    items: list[tuple[np.ndarray, str, str]] = []

    if os.path.isdir(args.path):
        for fn in sorted(os.listdir(args.path)):
            parsed = _parse_batch_filename(fn)
            if parsed is None:
                print(f"  skip {fn!r} — name doesn't match <char>_<form>_*.<ext>")
                continue
            ch, fm = parsed
            img = _load_image_gray(os.path.join(args.path, fn))
            glyph = _normalize_glyph(img)
            items.append((glyph, ch, fm))
            print(f"  loaded {fn!r:40s} → ('{ch}', '{fm}')")
    else:
        if not args.char or not args.form:
            sys.exit("ERROR: single-file mode needs `char` and `form` args.")
        img = _load_image_gray(args.path)
        glyph = _normalize_glyph(img)
        items.append((glyph, args.char, args.form))
        print(f"  loaded {args.path!r} → ('{args.char}', '{args.form}')")

    append_to_bank(items)


if __name__ == '__main__':
    main()
