"""Dump every glyph in the bank as a grid PNG, grouped by (letter, form).

Lets us eyeball whether the form labels in arabic_handwritten_dataset.npz
actually match the visual form. If a sample tagged Isolated looks like
Initial, the composer's `Initial fallback to Isolated` was *correct on the
labels* but wrong visually — and we'd want to relabel.

Usage:
    python src/show_glyph_bank.py                # all letters
    python src/show_glyph_bank.py --letter ب     # one letter, every sample
"""
from __future__ import annotations
import argparse, sys, io
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import paths

# Pick an Arabic-capable font for the titles.
for name in ('Arial', 'Tahoma', 'Segoe UI', 'Microsoft Sans Serif',
             'DejaVu Sans'):
    try:
        fm.findfont(name, fallback_to_default=False)
        matplotlib.rcParams['font.family'] = name
        break
    except Exception:
        continue


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('--letter', default=None,
                   help='If set, render every sample of this letter '
                        '(across all forms). Otherwise render all letters.')
    p.add_argument('--out', default=None,
                   help='Output PNG path (default: outputs/figures/'
                        'glyph_bank[_<letter>].png).')
    args = p.parse_args()

    d = np.load(paths.GLYPH_BANK_NPZ, allow_pickle=True)
    images = d['images']
    if images.ndim == 4:
        images = images[..., 0]
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)
    chars = [str(c) for c in d['char_labels']]
    forms = [str(f) for f in d['form_labels']]
    print(f"Bank: {len(chars)} samples, "
          f"{len(set(zip(chars, forms)))} unique (char,form) combos")

    # Group sample indices by (char, form).
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (c, f) in enumerate(zip(chars, forms)):
        groups[(c, f)].append(i)

    if args.letter:
        # One letter, every sample, grouped by form.
        forms_present = sorted({f for c, f in groups if c == args.letter})
        if not forms_present:
            print(f"Letter {args.letter!r} not in the bank.")
            return
        cells = []
        for fm_name in ('Isolated', 'Initial', 'Medial', 'Final'):
            for idx in groups.get((args.letter, fm_name), []):
                cells.append((idx, fm_name))
        title = f"Glyph bank — letter {args.letter!r} ({len(cells)} samples)"
        out = args.out or f"{paths.FIGURES_DIR}/glyph_bank_{args.letter}.png"
    else:
        # One representative cell per (char, form), in alphabetical order.
        cells = []
        for c, f in sorted(groups):
            cells.append((groups[(c, f)][0], f, c))
        cells = [(idx, f"{c}\n{f}") for idx, f, c in cells]
        title = f"Glyph bank — one sample per (letter, form), {len(cells)} cells"
        out = args.out or f"{paths.FIGURES_DIR}/glyph_bank.png"

    cols = 8
    rows = (len(cells) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes_flat = axes.flat if rows > 1 else axes
    for ax, cell in zip(axes_flat, cells):
        idx, label = cell
        ax.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        ax.set_title(label, fontsize=9)
        ax.axis('off')
    for ax in list(axes_flat)[len(cells):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = paths.ensure_dir(out)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
