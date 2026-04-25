"""Word composer — glue handwritten letter glyphs into a connected word image.

Input:  a real Arabic word (Unicode string) + a GlyphBank.
Output: one grayscale word image + the matching label (the input word).

Key operations, in order:
  1. Pick correct positional form per letter (PAW-aware, via glyph_bank.pick_form).
  2. Sample an augmented glyph per letter.
  3. Tight-crop each glyph to its ink bounding box (remove whitespace padding).
  4. Estimate each glyph's baseline = bottom of main letter body, ignoring
     descenders like ج ح خ ع غ م (they drop below the baseline).
  5. Paste all glyphs onto a fixed-height canvas, aligning baselines.
  6. Between connecting letters: overlap by a few pixels + draw a thin kashida
     at the baseline so letters appear joined.
  7. Between non-connecting letters (after ا د ذ ر ز و ة): leave a small gap.

Convention:
  - Glyphs are stacked LEFT→RIGHT in array space in logical order (first letter
    of the word on the left). Arabic reads RTL, so the rendered image will look
    "mirrored" to a human reader — but this matches the label order, which is
    what CTC wants. For real RTL data, horizontally flip once before feeding
    to the model.
"""

from __future__ import annotations
import numpy as np
import cv2

from glyph_bank import GlyphBank, pick_form, NON_CONNECTORS, CONNECTOR_SIDES


# ── Canvas / geometry defaults ────────────────────────────────────────────────
CANVAS_H = 48                # target image height
MARGIN_TOP = 6               # whitespace above the tallest ascender
BASELINE_FRAC = 0.70         # baseline sits at 70% of canvas height
CONNECT_OVERLAP = 3          # px horizontal overlap between edge-aligned letters
GAP_BETWEEN_PAWS = 3         # px gap after a non-connector (new sub-word)
EDGE_BAND = 3                # columns scanned at each edge to detect ink
INK_THRESHOLD = 128          # pixel < this = ink
MAX_TOTAL_H = 32             # cap on per-glyph total height — prevents the
                             # "small letter joined to huge letter" mismatch

# Letters with descenders — their lowest ink is BELOW the baseline, so we
# shouldn't use image-bottom as baseline for them.
DESCENDERS = set('جحخعغمهيىقص')


# ── Glyph post-processing ─────────────────────────────────────────────────────
def _tight_crop(glyph: np.ndarray, ink_threshold: int = 128) -> np.ndarray:
    """Crop the white padding around the letter so we have its true extent."""
    mask = glyph < ink_threshold
    if not mask.any():
        return glyph                             # all white, bail
    ys, xs = np.where(mask)
    return glyph[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


def _estimate_baseline(glyph: np.ndarray, letter: str,
                       ink_threshold: int = 128) -> int:
    """Y-coordinate (within the glyph) of the baseline.

    For non-descenders: baseline ≈ bottom of ink.
    For descenders: baseline ≈ 75% down (body sits above, tail below).
    """
    mask = glyph < ink_threshold
    if not mask.any():
        return glyph.shape[0] - 1
    if letter in DESCENDERS:
        # Find the widest horizontal band — that's the body; its bottom is
        # the baseline. Fast heuristic: baseline at 75% of glyph height.
        return int(glyph.shape[0] * 0.75)
    # Non-descender: baseline = bottom-most ink row.
    ys = np.where(mask.any(axis=1))[0]
    return int(ys.max())


def _normalize_height(glyph: np.ndarray, target_body_h: int,
                      letter: str, max_total_h: int | None = None) -> np.ndarray:
    """Scale glyph so its body is ~target_body_h tall AND total ≤ max_total_h.

    Two-step pass:
      1. Scale so body-above-baseline = target_body_h (consistent x-height).
      2. If total glyph height still exceeds max_total_h, shrink uniformly so
         the total fits — this caps tall ascenders and long descenders that
         would otherwise dwarf neighboring letters in the composed word.

    Without (2), a letter like ك (initial) with a tall vertical bar would end
    up ~35 px tall while a neighboring ت stays at ~22 px, and the joined word
    looks like the two letters live in different fonts.
    """
    baseline = _estimate_baseline(glyph, letter)
    body_h = baseline + 1
    if body_h <= 0:
        return glyph
    scale = target_body_h / body_h
    new_w = max(1, int(round(glyph.shape[1] * scale)))
    new_h = max(1, int(round(glyph.shape[0] * scale)))
    out = cv2.resize(glyph, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if max_total_h is not None and out.shape[0] > max_total_h:
        cap = max_total_h / out.shape[0]
        out = cv2.resize(out, (max(1, int(round(out.shape[1] * cap))),
                               max_total_h), interpolation=cv2.INTER_AREA)
    return out


# ── Kashida tip detection (form-topology aware, pixel-precise) ───────────────
def _kashida_tip(glyph: np.ndarray, side: str, baseline_y: int) -> dict:
    """Find the actual (x, y) tip of the kashida ink on one side of the glyph.

    Why this matters: the connector between two letters must start where the
    SOURCE letter's kashida ink physically ENDS, not at the bounding-box edge.
    If the kashida tail ends at column 2 of a glyph's box, drawing a connector
    from column 0 leaves a 2-px white sliver between the tail and the bar —
    the "letters touch but black ink doesn't" failure mode.

    Algorithm: scan rows in a baseline band; for each row, find the extremal
    ink column (leftmost on the left side, rightmost on the right side); the
    tip is the row whose extremum reaches farthest outward. Thickness is the
    vertical extent of the contiguous ink run at that column.

    Returns:
      x       — column inside the glyph where the kashida ink ends
      y       — row inside the glyph at the kashida tip
      thick   — vertical thickness of the kashida stroke at the tip
      present — False if no ink in the baseline band at all
    """
    h, w = glyph.shape
    y_lo = max(0, baseline_y - 8)
    y_hi = min(h, baseline_y + 7)
    band = glyph[y_lo:y_hi]                     # shape (band_h, w)
    mask = band < INK_THRESHOLD

    if not mask.any():
        return dict(x=(0 if side == 'left' else w - 1),
                    y=baseline_y, thick=2, present=False)

    # For each row in the band, find the extremal ink column on this side.
    row_extreme = np.full(mask.shape[0], -1 if side == 'right' else w,
                          dtype=np.int32)
    for r in range(mask.shape[0]):
        row_mask = mask[r]
        if not row_mask.any():
            continue
        idxs = np.where(row_mask)[0]
        row_extreme[r] = int(idxs[-1] if side == 'right' else idxs[0])

    if side == 'right':
        tip_row_in_band = int(np.argmax(row_extreme))
        tip_x = int(row_extreme[tip_row_in_band])
        if tip_x < 0:
            return dict(x=w - 1, y=baseline_y, thick=2, present=False)
    else:
        tip_row_in_band = int(np.argmin(row_extreme))
        tip_x = int(row_extreme[tip_row_in_band])
        if tip_x >= w:
            return dict(x=0, y=baseline_y, thick=2, present=False)

    tip_y = y_lo + tip_row_in_band

    # Estimate thickness at the tip by counting contiguous ink rows in a
    # 3-column window around tip_x at tip_y.
    col_lo, col_hi = max(0, tip_x - 1), min(w, tip_x + 2)
    col_mask = (glyph[:, col_lo:col_hi] < INK_THRESHOLD).any(axis=1)
    up = tip_y
    while up > 0 and col_mask[up - 1]:
        up -= 1
    down = tip_y
    while down < h - 1 and col_mask[down + 1]:
        down += 1
    thick = max(1, down - up + 1)

    return dict(x=tip_x, y=tip_y, thick=thick, present=True)


def _draw_kashida(canvas: np.ndarray, x_from: int, x_to: int, y: int,
                  thickness: int, ink_value: int = 8) -> None:
    """Draw a HORIZONTAL kashida segment at row `y` between x_from and x_to.

    Always horizontal (baseline-locked), uses dark ink (value 8) close to
    the glyph stroke darkness so the connector merges visually into the
    existing letter ink instead of looking like a separate gray bar.

    The endpoints are inclusive — we extend the segment by 1 px on each end
    so it actually overlaps the existing ink at the kashida tips and forms
    one continuous stroke.
    """
    if x_from == x_to:
        return
    a, b = (x_from, x_to) if x_from < x_to else (x_to, x_from)
    a = max(0, a - 1)
    b = min(canvas.shape[1] - 1, b + 1)
    half = max(0, thickness // 2)
    y0 = max(0, y - half)
    y1 = min(canvas.shape[0], y + half + 1)
    region = canvas[y0:y1, a:b + 1]
    np.minimum(region, np.uint8(ink_value), out=region)


# ── Composition (edge-aware) ──────────────────────────────────────────────────
def compose_word(word: str, bank: GlyphBank,
                 canvas_h: int = CANVAS_H,
                 target_body_h: int = 22) -> np.ndarray:
    """Render one Arabic word with kashidas locked to a single canvas baseline.

    Two ideas drive the composition:

    (a) FORM TOPOLOGY IS GROUND TRUTH.
        Each positional form declares which sides carry a kashida (CONNECTOR_SIDES
        in glyph_bank). We use the *resolved* form (after the topology-preserving
        fallback in GlyphBank._resolve), so we always know which edges should
        carry a connector — instead of guessing from pixel content.

    (b) CONNECTORS LIVE ON THE CANVAS BASELINE, NOT THE GLYPH BASELINE.
        Per-glyph baselines drift (descenders, body height, augmentation jitter);
        if you connect at per-glyph y-centers, letters stack at slightly different
        heights and the kashida slants. Instead, we Y-shift each glyph so its
        kashida-side ink sits at the canvas baseline_y row, then draw the kashida
        as a strict horizontal segment between consecutive letters at that row.
    """
    baseline_y = int(canvas_h * BASELINE_FRAC)

    # ── Phase 1: prep each glyph ──────────────────────────────────────────────
    prepared: list[dict] = []
    for i, ch in enumerate(word):
        form_req = pick_form(word, i)
        form_used = bank.resolve_form(ch, form_req) or form_req
        g = bank.sample(ch, form_req)
        g = _tight_crop(g)
        g = _normalize_height(g, target_body_h, ch, max_total_h=MAX_TOTAL_H)

        is_last = i == len(word) - 1
        connects_forward  = ch not in NON_CONNECTORS and not is_last
        connects_backward = (i > 0) and (word[i - 1] not in NON_CONNECTORS)

        # The form's declared kashida topology — what we EXPECT in the glyph.
        carry_left, carry_right = CONNECTOR_SIDES.get(form_used, (False, False))
        bl = _estimate_baseline(g, ch)
        left_k  = _kashida_tip(g, 'left',  bl) if carry_left  else None
        right_k = _kashida_tip(g, 'right', bl) if carry_right else None

        prepared.append({
            'char': ch,
            'form_used': form_used,
            'glyph': g,
            'baseline': bl,
            'left_k': left_k,        # connector point at left edge (or None)
            'right_k': right_k,      # connector point at right edge (or None)
            'connects_forward': connects_forward,
            'connects_backward': connects_backward,
        })

    # ── Phase 2: paste RTL on the canvas (word[0] ends up at the right edge) ──
    canvas_w = sum(p['glyph'].shape[1] for p in prepared) + len(prepared) * 4 + 16
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    cursor_right = canvas_w - 2

    # Track the PREVIOUS letter's left-kashida tip in CANVAS coordinates so the
    # next letter can dock to it: we want
    #     x_start_B + tip_x_B_right  ==  prev_tip_x          (same canvas col)
    #     baseline_y + tip_y_B_right ==  baseline_y          (same canvas row)
    # i.e. THIS letter's right-tip pixel sits on TOP of the previous letter's
    # left-tip pixel — one continuous stroke, no separate connector bar needed.
    prev_tip_x: int | None = None
    prev_thickness = 2

    for p in prepared:
        g = p['glyph']
        gh, gw = g.shape

        # Y-shift: snap the kashida-side ink to canvas baseline_y so the dock
        # row matches between letters. (Right-kashida wins when both exist —
        # the back-connection is what we're docking right now.)
        if p['connects_backward'] and p['right_k'] and p['right_k']['present']:
            top = baseline_y - p['right_k']['y']
        elif p['connects_forward'] and p['left_k'] and p['left_k']['present']:
            top = baseline_y - p['left_k']['y']
        else:
            top = baseline_y - p['baseline']

        # X-place: when this letter docks back, solve for x_start so the
        # right-tip of THIS letter coincides with prev letter's left-tip.
        # Otherwise (PAW start, or no kashida), fall back to the cursor.
        if p['connects_backward'] and prev_tip_x is not None \
                and p['right_k'] and p['right_k']['present']:
            x_start = prev_tip_x - p['right_k']['x']
        else:
            x_start = cursor_right - gw

        # Composite glyph onto canvas (dark-wins). The two letters' kashida ink
        # already overlap exactly at the dock pixel, so there's no need for a
        # separate connector — but we keep a 1-px-wide bridge below in case
        # the kashida runs are short enough to leave a hairline gap inside the
        # baseline band.
        bot = top + gh
        src_top = max(0, -top)
        src_bot = gh - max(0, bot - canvas_h)
        dst_top = max(0, top)
        dst_bot = min(canvas_h, bot)
        if dst_bot > dst_top and src_bot > src_top and 0 <= x_start \
                and x_start + gw <= canvas_w:
            region = canvas[dst_top:dst_bot, x_start:x_start + gw]
            src = g[src_top:src_bot]
            if region.shape == src.shape:
                np.minimum(region, src, out=region)

        # Safety bridge: paint a short, dark, baseline-locked segment between
        # the two tips. With exact docking the two ink runs already overlap,
        # so this bar is usually 0–2 px wide — but it guarantees pixel
        # continuity even if augmentation eroded one of the kashida ends.
        if p['connects_backward'] and prev_tip_x is not None \
                and p['right_k'] and p['right_k']['present']:
            entry_x = x_start + p['right_k']['x']
            thick = max(2, (prev_thickness + p['right_k']['thick']) // 2)
            _draw_kashida(canvas, prev_tip_x, entry_x, baseline_y, thick)

        # Record THIS letter's left-kashida tip for the NEXT iteration.
        if p['connects_forward'] and p['left_k'] and p['left_k']['present']:
            prev_tip_x = x_start + p['left_k']['x']
            prev_thickness = p['left_k']['thick']
            cursor_right = x_start                    # next letter docks via prev_tip_x
        else:
            prev_tip_x = None
            prev_thickness = 2
            cursor_right = x_start - GAP_BETWEEN_PAWS  # visible PAW gap

    # Trim outer whitespace.
    mask = canvas < 255
    if mask.any():
        xs = np.where(mask.any(axis=0))[0]
        canvas = canvas[:, max(0, xs.min() - 2): xs.max() + 3]

    return canvas


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, io, os
    import paths
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    bank = GlyphBank(npz_path=paths.GLYPH_BANK_NPZ, seed=7)
    words = [
        ('كتاب',  'w1_kitab'),
        ('مدرسة', 'w2_madrasa'),
        ('كرسي',  'w3_kursi'),
        ('وردة',  'w4_warda'),
        ('بيت',   'w5_bait'),
        ('محمد',  'w6_muhammad'),
        ('سلام',  'w7_salam'),
        ('علم',   'w8_ilm'),
    ]
    out_dir = f"{paths.OUTPUTS_DIR}/compose_test"
    os.makedirs(out_dir, exist_ok=True)
    # Clean stale files from prior runs.
    for fn in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, fn))

    for w, stem in words:
        try:
            img = compose_word(w, bank)
        except KeyError as e:
            print(f"  {w:10s} SKIP: {e}")
            continue
        out = f'{out_dir}/{stem}.png'
        cv2.imwrite(out, img)
        print(f"  {w:10s} {img.shape} → {out}")
