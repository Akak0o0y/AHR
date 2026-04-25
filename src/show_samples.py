"""Render a grid of synth samples with their labels + CRNN predictions.

Saves a PNG (samples_grid.png) so the user can eyeball:
  - what the synth handwriting looks like
  - the CRNN's predicted decoding next to the ground truth
"""
from __future__ import annotations
import sys, io, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from solution import decode_prediction, IDX_TO_CHAR, ctc_loss_fn
from train import load_corpus
import paths

# Try to find an Arabic-capable font on the system; fall back to default if none
import matplotlib.font_manager as fm
ARABIC_FONTS = ['Arial', 'Tahoma', 'Segoe UI', 'Times New Roman',
                'Microsoft Sans Serif', 'DejaVu Sans']
for name in ARABIC_FONTS:
    try:
        fm.findfont(name, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = name
        break
    except Exception:
        continue


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=7,
                   help='RNG seed for which samples to draw.')
    p.add_argument('--n', type=int, default=24,
                   help='Number of samples to render (must equal cols*rows).')
    p.add_argument('--out', default=None,
                   help=f'Output PNG path. Default: {paths.SAMPLES_GRID_PNG} '
                        '(or _seed<N> suffix when --seed != 7).')
    args = p.parse_args()

    n_show = args.n
    if args.out:
        out_path = args.out
    elif args.seed == 7:
        out_path = paths.SAMPLES_GRID_PNG
    else:
        stem, ext = paths.SAMPLES_GRID_PNG.rsplit('.', 1)
        out_path = f"{stem}_seed{args.seed}.{ext}"
    out_path = paths.ensure_dir(out_path)

    d = np.load(paths.SYNTH_CORPUS_ENRICHED, allow_pickle=True)
    images_raw, labels_raw = d['images'], d['labels']
    en_defs = d['en_definitions']
    print(f"Loaded {len(images_raw)} samples")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(images_raw), size=n_show, replace=False)

    # Get matching prepared inputs + predictions
    X, y, _ = load_corpus(paths.SYNTH_CORPUS_ENRICHED)
    model = tf.keras.models.load_model(
        paths.CRNN_MODEL, custom_objects={'ctc_loss_fn': ctc_loss_fn})
    preds = model.predict(X[idx], verbose=0)

    cols, rows = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.0))
    for ax, i, p in zip(axes.flat, idx, preds):
        ax.imshow(images_raw[i], cmap='gray', vmin=0, vmax=255)
        true = str(labels_raw[i])
        pred = decode_prediction(p)
        en = (en_defs[i] or '')[:24]
        marker = '✓' if pred == true else '✗'
        # NOTE: matplotlib renders Arabic LTR (no shaping/bidi), but characters
        # still display correctly in isolated form — fine for this sanity peek.
        ax.set_title(f"{marker} true: {true}\npred: {pred}\nen: {en}",
                     fontsize=8, loc='left')
        ax.axis('off')
    fig.suptitle(f"Synth corpus samples + CRNN predictions "
                 f"({n_show} random, seed={args.seed})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f"Saved → {out_path}")

    # Also dump a text table so the Arabic renders correctly in a terminal
    print(f"\n=== {n_show} random samples (pred / true / English gloss) ===")
    for i, p in zip(idx, preds):
        true = str(labels_raw[i])
        pred = decode_prediction(p)
        en = (en_defs[i] or '')[:60]
        marker = '✓' if pred == true else '✗'
        print(f"  {marker}  true={true!r:14s}  pred={pred!r:14s}  en={en}")


if __name__ == '__main__':
    main()
