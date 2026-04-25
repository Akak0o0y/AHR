"""Synthetic word corpus — turns an Arabic word list into (image, label) pairs.

Pipeline:
  word list (text file, one word per line)
    → filter to words whose letters are all covered by the glyph bank
    → for each word, generate N synthetic images (each with fresh augmentation)
    → save as .npz (images, labels) ready to feed into CRNN training

Word list sources (try in order):
  1. User-supplied path (--word_list).
  2. Bundled tiny starter list (WORD_LIST_STARTER) — ~150 common Arabic words,
     enough to smoke-test the full pipeline end-to-end without downloads.

For production, grab:
  - Hunspell ar_SA dictionary (~100k words) — https://github.com/wooorm/dictionaries
  - Tashkeela corpus (~75M words, subset to unique forms)
  - Arabic Wikipedia vocab via `arabic-corpora` package

Image format:
  - Variable width, fixed height (CANVAS_H from word_composer).
  - Stored as uint8 grayscale arrays in a list (object dtype in npz).
  - Labels are the original logical-order Unicode strings.
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Iterable
import numpy as np
from tqdm import tqdm

from glyph_bank import GlyphBank, pick_form
from word_composer import compose_word
from dictionary import ArabicDictionary
import paths


# ── Tiny starter word list (for smoke tests; swap with hunspell for training) ──
# 150 common Modern Standard Arabic words. Filtered to use letters our glyph
# bank actually covers (no hamza variants requiring glyphs we don't have).
WORD_LIST_STARTER = """
كتاب مدرسة بيت باب ولد بنت رجل امرأة صديق عمل
طالب معلم جامعة درس قلم كرسي طاولة سرير غرفة مطبخ
ماء طعام خبز لحم سمك فاكهة خضار شاي قهوة حليب
شمس قمر نجم سماء أرض بحر نهر جبل شجرة وردة
كلب قط حصان طائر سيارة طريق مدينة قرية دولة وطن
حب سلام حرب عدل ظلم خير شر نور ظلام يوم
ليل صباح مساء أسبوع شهر سنة وقت ساعة دقيقة ثانية
كبير صغير طويل قصير قوي ضعيف سريع بطيء جميل قبيح
ذهب فضة حديد نحاس نار ريح رمل ثلج مطر برق
رأس عين أذن أنف فم يد قدم قلب عقل روح
نوم طعام شراب علم فهم ذكاء صبر شجاعة صدق كذب
واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة
أحمر أزرق أخضر أصفر أبيض أسود بني رمادي برتقالي بنفسجي
كتب قرأ درس فهم سمع شاهد ركض مشى أكل شرب
سأل جواب فكر رسم غنى رقص ضحك بكى نام استيقظ
""".split()


def load_word_list(path: str | None) -> list[str]:
    """Load words from file, or fall back to bundled starter list."""
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            words = [w.strip() for w in f if w.strip()]
        print(f"Loaded {len(words)} words from {path}")
        return words
    print("No word list provided — using bundled starter list (~150 words)")
    return WORD_LIST_STARTER


def filter_renderable(words: Iterable[str], bank: GlyphBank,
                      dictionary: ArabicDictionary | None = None,
                      require_dict: bool = False,
                      verbose: bool = False) -> list[str]:
    """Keep only words where every letter resolves to a bank glyph.

    If `require_dict=True`, also drops words with no entry in `dictionary` —
    useful when you only want corpus samples that can carry definitions.
    """
    kept, dropped = [], []
    for w in words:
        if len(w) < 2 or len(w) > 10:
            dropped.append((w, 'length'))
            continue
        try:
            for i, ch in enumerate(w):
                if not bank.has(ch, pick_form(w, i)):
                    raise KeyError(ch)
        except KeyError as e:
            dropped.append((w, f'no glyph for {e}'))
            continue
        if require_dict and dictionary is not None and w not in dictionary:
            dropped.append((w, 'no dict entry'))
            continue
        kept.append(w)
    print(f"Renderable: {len(kept)}/{len(kept) + len(dropped)} words")
    if verbose and dropped:
        for w, reason in dropped[:10]:
            print(f"  dropped: {w!r} — {reason}")
    return kept


def generate_corpus(words: list[str], bank: GlyphBank,
                    samples_per_word: int = 20,
                    out_path: str = 'synth_corpus.npz',
                    dictionary: ArabicDictionary | None = None) -> None:
    """Render `samples_per_word` variants of each word; save to .npz.

    If `dictionary` is given, the .npz also contains `ar_definitions` and
    `en_definitions` arrays aligned with `labels` — empty strings for words
    not found in the dictionary. Training stays letter-only (train.py ignores
    these extra fields); they're metadata for later image→meaning models.
    """
    images: list[np.ndarray] = []
    labels: list[str] = []
    ar_defs: list[str] = []
    en_defs: list[str] = []
    pos_labels: list[str] = []

    for w in tqdm(words, desc='Synthesizing'):
        ar  = dictionary.ar_def(w)  if dictionary else ''
        en  = dictionary.en_def(w)  if dictionary else ''
        pos = dictionary.pos_def(w) if dictionary else ''
        for _ in range(samples_per_word):
            try:
                img = compose_word(w, bank)
            except KeyError:
                break  # word unrenderable (shouldn't happen after filter)
            images.append(img)
            labels.append(w)
            ar_defs.append(ar)
            en_defs.append(en)
            pos_labels.append(pos)

    print(f"\nGenerated {len(images)} samples across {len(set(labels))} unique words")
    if dictionary:
        n_with_any = sum(1 for i in range(len(labels))
                         if ar_defs[i] or en_defs[i])
        n_pos = sum(1 for p in pos_labels if p)
        print(f"Dictionary coverage: {n_with_any}/{len(labels)} samples have "
              f"at least one definition "
              f"({sum(1 for d in ar_defs if d)} ar, "
              f"{sum(1 for d in en_defs if d)} en, "
              f"{n_pos} pos)")

    # Save: object-dtype arrays since image widths vary. Pre-allocate so numpy
    # doesn't try to broadcast ragged shapes into a regular array.
    img_arr = np.empty(len(images), dtype=object)
    for i, im in enumerate(images):
        img_arr[i] = im
    save_kwargs = {'images': img_arr, 'labels': np.array(labels)}
    if dictionary:
        save_kwargs['ar_definitions'] = np.array(ar_defs)
        save_kwargs['en_definitions'] = np.array(en_defs)
        save_kwargs['pos_labels']     = np.array(pos_labels)
    np.savez_compressed(out_path, **save_kwargs)
    print(f"Saved → {out_path}")

    # Quick stats.
    widths = [img.shape[1] for img in images]
    print(f"Image widths: min={min(widths)}, max={max(widths)}, "
          f"mean={np.mean(widths):.0f}, median={int(np.median(widths))}")


def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    p = argparse.ArgumentParser()
    p.add_argument('--word_list', default=None,
                   help='Path to word list file (one word per line). '
                        'Uses bundled starter list if omitted.')
    p.add_argument('--glyph_bank', default=paths.GLYPH_BANK_NPZ)
    p.add_argument('--samples_per_word', type=int, default=20)
    p.add_argument('--out', default=paths.SYNTH_CORPUS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--dictionary', default=None,
                   help='Path to flat JSON {word: {ar, en, pos}}. '
                        'When set, the .npz gains ar_definitions and '
                        'en_definitions arrays aligned with labels.')
    p.add_argument('--require_dict', action='store_true',
                   help='Drop words that are not in the dictionary.')
    p.add_argument('--from_dict', action='store_true',
                   help='Use ALL Arabic-script keys in --dictionary as the '
                        'word list (ignores --word_list). The corpus then '
                        'covers every renderable headword the dictionary '
                        'knows about, with full ar/en/pos metadata attached.')
    args = p.parse_args()

    bank = GlyphBank(npz_path=args.glyph_bank, seed=args.seed)
    dictionary = ArabicDictionary(args.dictionary) if args.dictionary else None
    if args.from_dict:
        if dictionary is None:
            sys.exit("--from_dict needs --dictionary.")
        # Iterate every dict key, keep Arabic-script ones (the dump also
        # contains a few stray English entries like 'overweight').
        words = [w for w in dictionary._entries.keys()
                 if w and all(0x0600 <= ord(c) <= 0x06FF for c in w)]
        print(f"Using {len(words)} Arabic-script keys from the dictionary "
              f"as the word list.")
    else:
        words = load_word_list(args.word_list)
    words = filter_renderable(words, bank,
                              dictionary=dictionary,
                              require_dict=args.require_dict,
                              verbose=args.verbose)

    if not words:
        print("ERROR: no renderable words after filter. Check glyph bank coverage.")
        sys.exit(1)

    generate_corpus(words, bank,
                    samples_per_word=args.samples_per_word,
                    out_path=paths.ensure_dir(args.out),
                    dictionary=dictionary)


if __name__ == '__main__':
    main()
