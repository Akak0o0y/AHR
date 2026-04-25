"""Glyph bank — indexed handwritten letter samples by (letter, positional form).

Loads from arabic_handwritten_dataset.npz (or a custom path), groups samples by
(character, form), and provides per-call augmentation so every synthetic word
sees a fresh-looking glyph even when only 1–2 source samples exist.

Augmentations per sample() call (all independent random per call):
  - elastic distortion  (simulates handwriting stroke jitter)
  - stroke thickness    (dilate / erode kernel 1px)
  - rotation            (±3°)
  - slight translation  (±1px)
  - brightness noise    (±10)

This is what keeps the pipeline from overfitting to the exact 1 sample we have
per (letter, form) — each composed word uses a unique per-glyph augmentation.
"""

from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import cv2


# The 6 Arabic non-connectors — these letters DO NOT connect to the next letter.
# After one of these, the next letter takes its Initial form (not Medial),
# and the non-connector itself only ever appears as Isolated or Final.
NON_CONNECTORS = set('اأإآدذرزوة')

# Forms we recognize. Isolated = used alone, not inside a connected run.
FORMS = ('Isolated', 'Initial', 'Medial', 'Final')

# Connector topology per form: (carries_left_kashida, carries_right_kashida)
# In our pre-RTL-flip image space:
#   - "left" edge of the glyph faces the NEXT letter in the word
#     (which sits to the LEFT in our LTR canvas before the global flip)
#   - "right" edge faces the PREVIOUS letter
# So an Initial letter (which connects only to the next letter) carries a
# left kashida; a Final letter (which connects only to the previous letter)
# carries a right kashida; Medial carries both; Isolated carries neither.
CONNECTOR_SIDES: dict[str, tuple[bool, bool]] = {
    'Isolated': (False, False),
    'Initial':  (True,  False),
    'Medial':   (True,  True),
    'Final':    (False, True),
}


def _form_fallback_chain(form: str) -> list[str]:
    """Forms whose kashida topology is COMPATIBLE with `form`.

    A fallback form is acceptable iff it carries kashidas on AT LEAST the
    sides `form` requires. This is what stops 'Initial ب is missing → use
    Isolated ب', because Isolated has no left kashida and the composer would
    have nothing to connect to.

    Example for form='Initial' (needs left kashida, no right):
      → ['Initial', 'Medial']   (Medial has left+right; extra right is a
                                 small artifact at word start, far better
                                 than the disconnected look of Isolated)
    Example for form='Isolated' (needs neither):
      → ['Isolated', 'Initial', 'Medial', 'Final']  (anything works)
    """
    need_l, need_r = CONNECTOR_SIDES[form]
    out = [form]                                                 # exact first
    for f, (l, r) in CONNECTOR_SIDES.items():
        if f == form:                       continue
        if (need_l and not l) or (need_r and not r): continue    # missing required kashida
        out.append(f)
    return out

# Visual-equivalent fallback chains. When a glyph for a letter is missing,
# try these substitutes (top of chain preferred). These cover the common gap:
# dictionary words use bare `ا` but annotators often label alifs as `أ`.
EQUIVALENTS: dict[str, tuple[str, ...]] = {
    'ا': ('ا', 'أ', 'إ', 'آ'),
    'أ': ('أ', 'ا', 'إ', 'آ'),
    'إ': ('إ', 'ا', 'أ'),
    'آ': ('آ', 'أ', 'ا'),
    'ى': ('ى', 'ي'),       # alif maqsura ≈ ya visually without dots
    'ة': ('ة', 'هـ', 'ه'),  # taa marbuta ≈ haa
    'ه': ('ه', 'هـ', 'ة'),  # bare ha ≈ ha-with-tatweel (dataset stores 'هـ')
    'ؤ': ('ؤ', 'و'),
    'ئ': ('ئ', 'ي'),
    'ء': ('ء', 'أ'),       # hamza on the line — no good substitute, try alif-hamza
}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation primitives
# ─────────────────────────────────────────────────────────────────────────────
def _elastic(img: np.ndarray, alpha: float = 8.0, sigma: float = 3.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Elastic deformation — simulates handwriting stroke wobble."""
    rng = rng or np.random.default_rng()
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1), (0, 0), sigma) * alpha
    map_x = (np.arange(w)[None, :] + dx).astype(np.float32)
    map_y = (np.arange(h)[:, None] + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _jitter_stroke(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly thicken or thin ink strokes by 1px."""
    kernel = np.ones((2, 2), np.uint8)
    op = rng.integers(0, 3)  # 0=none, 1=dilate, 2=erode
    if op == 1:
        # Ink is dark on light bg → dilating ink means eroding the image.
        return cv2.erode(img, kernel, iterations=1)
    if op == 2:
        return cv2.dilate(img, kernel, iterations=1)
    return img


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255,
                          flags=cv2.INTER_LINEAR)


def _translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          borderValue=255, flags=cv2.INTER_LINEAR)


def augment(img: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Full augmentation stack applied per sample() call."""
    rng = rng or np.random.default_rng()
    out = img.copy()
    if rng.random() < 0.8:
        out = _elastic(out, alpha=rng.uniform(4, 10), sigma=rng.uniform(2.5, 4.0), rng=rng)
    if rng.random() < 0.5:
        out = _jitter_stroke(out, rng)
    if rng.random() < 0.7:
        out = _rotate(out, rng.uniform(-3, 3))
    if rng.random() < 0.4:
        out = _translate(out, int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
    if rng.random() < 0.3:
        noise = rng.integers(-10, 10, out.shape, dtype=np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Glyph bank
# ─────────────────────────────────────────────────────────────────────────────
class GlyphBank:
    """Lookup handwritten letter samples by (letter, form) with augmentation."""

    def __init__(self, npz_path: str = 'arabic_handwritten_dataset.npz',
                 seed: int | None = None):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"{npz_path} not found. Run export_labeled_dataset() first.")
        data = np.load(npz_path, allow_pickle=True)
        self.rng = np.random.default_rng(seed)
        self._bank: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        images = data['images']
        if images.ndim == 4:           # (N, H, W, 1) → (N, H, W)
            images = images[..., 0]
        if images.dtype != np.uint8:   # augmented npz stored as float [0,1]
            images = (images * 255).astype(np.uint8)
        for img, char, form in zip(images, data['char_labels'], data['form_labels']):
            self._bank[(str(char), str(form))].append(img)
        print(f"GlyphBank: {sum(len(v) for v in self._bank.values())} samples, "
              f"{len(self._bank)} unique (char,form) combos loaded from {npz_path}")

    # ── Lookup ────────────────────────────────────────────────────────────────
    def has(self, letter: str, form: str) -> bool:
        """True if any fallback (exact, Isolated, or visual-equiv) is reachable."""
        return self._resolve(letter, form) is not None

    def _resolve(self, letter: str, form: str) -> tuple[str, str] | None:
        """Find the first available (letter_variant, form_variant) in the bank.

        Two-pass cascade:
          Pass 1 — topology-preserving: try `form` exactly, then any other
                   form whose kashida topology is COMPATIBLE (carries the
                   required kashidas). Walk the EQUIVALENTS chain inside this
                   pass so a visually-equivalent letter with the right form
                   beats the same letter with a disconnected form.
          Pass 2 — last resort: any form, any equivalent (may produce visible
                   disconnection — fires only when the bank is missing every
                   kashida-compatible form for every visual equivalent).
        """
        chain = EQUIVALENTS.get(letter, (letter,))

        # Pass 1: kashida-topology preserved.
        for fv in _form_fallback_chain(form):
            for lv in chain:
                if (lv, fv) in self._bank:
                    return (lv, fv)

        # Pass 2: last resort, accept disconnection as visible artifact.
        all_forms = [form] + [f for f in FORMS if f != form]
        for lv in chain:
            for fv in all_forms:
                if (lv, fv) in self._bank:
                    return (lv, fv)
        return None

    def resolve_form(self, letter: str, form: str) -> str | None:
        """Public: return the form actually used after fallback (or None)."""
        key = self._resolve(letter, form)
        return key[1] if key else None

    def sample(self, letter: str, form: str,
               augment_on: bool = True) -> np.ndarray:
        """Return one augmented 32x32 uint8 glyph for (letter, form).

        Uses form-level and visual-equivalent fallbacks when the exact
        (letter, form) is missing.
        """
        key = self._resolve(letter, form)
        if key is None:
            raise KeyError(f"No glyph for '{letter}' in any form or equivalent")
        img = self._bank[key][self.rng.integers(0, len(self._bank[key]))]
        return augment(img, self.rng) if augment_on else img.copy()

    def coverage_report(self) -> dict:
        """What (letter, form) combos exist? For sanity-checking."""
        return {k: len(v) for k, v in sorted(self._bank.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Positional form picker — the PAW-aware logic
# ─────────────────────────────────────────────────────────────────────────────
def pick_form(word: str, i: int) -> str:
    """Return the correct positional form for word[i].

    Rules:
      - Single-letter word             → Isolated
      - Current letter is non-connector → Isolated (if first/after non-connector)
                                          Final    (if connects to previous)
      - First letter, or previous was a non-connector → Initial
      - Last letter (and previous connects)           → Final
      - Otherwise                                     → Medial
    """
    if len(word) == 1:
        return 'Isolated'

    ch = word[i]
    prev_connects = (i > 0) and (word[i - 1] not in NON_CONNECTORS)

    if ch in NON_CONNECTORS:
        # Non-connector: never has Initial/Medial. Isolated if standalone start,
        # else Final (it closes the preceding connected run).
        return 'Final' if prev_connects else 'Isolated'

    is_last = (i == len(word) - 1)
    if not prev_connects and is_last:
        return 'Isolated'
    if not prev_connects:
        return 'Initial'
    if is_last:
        return 'Final'
    return 'Medial'


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys, io
    import paths
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    bank = GlyphBank(npz_path=paths.GLYPH_BANK_NPZ, seed=42)

    # Show form picker correctness on known-tricky words.
    tests = ['كتاب', 'مدرسة', 'كرسي', 'وردة', 'بيت', 'ا']
    for w in tests:
        forms = [pick_form(w, i) for i in range(len(w))]
        print(f"{w:10s} → {list(zip(w, forms))}")

    # Sanity: sample one glyph per letter of كتاب and show coverage.
    word = 'كتاب'
    missing = []
    for i, ch in enumerate(word):
        f = pick_form(word, i)
        if not bank.has(ch, f) and not bank.has(ch, 'Isolated'):
            missing.append((ch, f))
    print(f"\n'{word}' — missing glyphs:", missing or 'none')

    # Write one augmented glyph strip for visual inspection.
    strip = np.hstack([bank.sample(ch, pick_form(word, i)) for i, ch in enumerate(word)])
    out_png = paths.ensure_dir(f"{paths.FIGURES_DIR}/_glyph_bank_test.png")
    cv2.imwrite(out_png, strip)
    print(f"Wrote {out_png} (left-to-right unconnected letters).")
