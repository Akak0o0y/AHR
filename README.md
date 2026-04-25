# Arabic Handwritten Recognition (AHR)

Segmentation-free Arabic handwriting OCR via CRNN + CTC, trained on
pixel-aware synthesized handwritten words bootstrapped from a small
single-writer letter bank.

> Full design rationale, architecture, theory, and roadmap live in
> [PROJECT.md](PROJECT.md).

## Layout

```
AHR/
├── PROJECT.md                  Canonical design doc (theory, rules, roadmap)
├── README.md                   This file (quickstart + layout)
│
├── data/
│   ├── raw/                    Source data (do not modify)
│   │   ├── arabic_handwritten_dataset.npz    Glyph bank (101 letter samples)
│   │   ├── arabic_handwritten_dataset.bak.npz Pre-Initial-ب-add backup
│   │   ├── ar_dictionary.json                Flat {word: {ar,en,pos}} dict (54k)
│   │   ├── new_glyphs/                       Drop-zone for new bank samples
│   │   └── real_test/                        Real handwritten test images
│   └── synth/                  Generated training corpora
│       ├── synth_corpus_dict.npz             52,821-sample dictionary corpus (canonical)
│       └── synth_corpus_5k.npz               15k-sample / 5000-word corpus
│
├── models/                     Trained model checkpoints
│   ├── expert_best.keras       Letter-form classifier (CNN backbone donor)
│   └── crnn_arabic_5k.keras    Word OCR CRNN (latest)
│
├── scripts/                    Essential scripts to run the model
│   ├── paths.py                Canonical path constants
│   ├── solution.py             CRNN architecture, alphabet, encode/decode, CTC loss
│   ├── glyph_bank.py           Letter-bank lookup with topology-preserving fallback
│   ├── word_composer.py        Pixel-aware RTL composition with kashida-tip docking
│   ├── synth_corpus.py         Word-list → synth corpus (with --from_dict mode)
│   ├── dictionary.py           Wiktionary fetcher + lookup
│   ├── add_to_bank.py          Append new (letter, form) samples to the bank
│   ├── train.py                CRNN training (no early stop, no val split)
│   ├── test_crnn.py            Accuracy + CER evaluation
│   └── predict.py              Single-image inference (image → word)
│
├── notebooks/                  Jupyter exploration + notebook utilities
│
└── outputs/
    ├── figures/                Sample grids, bank visualizations, training plots
    ├── compose_test/           word_composer.py smoke-test renders
    └── logs/                   Training and dictionary-fetch logs
```

## Quickstart

All scripts use absolute paths anchored at this repo, so they work from any cwd:

```bash
# 1. (One-time) Build the dictionary — ~40 MB gzipped download, 55k entries
python scripts/dictionary.py fetch

# 2. Generate synth corpus from EVERY renderable dictionary entry (1 sample/word)
python scripts/synth_corpus.py --from_dict --dictionary data/raw/ar_dictionary.json \
    --samples_per_word 1 --out data/synth/synth_corpus_dict.npz

# 3. Smoke-train the CRNN (1 epoch sanity check)
python scripts/train.py --smoke --corpus data/synth/synth_corpus_dict.npz

# 4. Full training (no early stop, ReduceLROnPlateau on train loss)
python scripts/train.py --corpus data/synth/synth_corpus_5k.npz --epochs 25

# 5. Evaluate accuracy + character error rate
python scripts/test_crnn.py

# 6. Inference on one image
python scripts/predict.py path/to/word.png
```

## Adding new letter samples to the bank

```bash
# Single file — explicit (letter, form):
python scripts/add_to_bank.py path/to/img.png ب Initial

# Or batch — filenames encode (letter, form), e.g. ب_Initial_01.png:
python scripts/add_to_bank.py data/raw/new_glyphs/
```

The original bank is auto-backed up to `*.bak.npz` before the first append.

## Path conventions

`scripts/paths.py` is the single source of truth for filesystem locations.
To relocate any artifact (e.g. point at a different glyph bank or a different
canonical model), edit `paths.py` rather than chasing string literals.

## Current results (latest model on the dictionary corpus)

`crnn_arabic_5k.keras` was trained on `synth_corpus_5k.npz` (15k images / 5000
unique words) for 25 epochs. Evaluated on the full 52,821-word
`synth_corpus_dict.npz`:

- Final training loss: **0.018**
- Exact-match accuracy on 52,821 words: **98.19 %** (≈47k of the test words
  were unseen during training — strong evidence the model learned letter
  shapes, not word memorization)
- Character Error Rate (CER): **0.36 %**

Real-world evaluation on photographed handwritten samples currently fails
(documented as the synth↔real domain gap; see PROJECT.md §3.13). Phase 4
fine-tuning on labeled real samples is the planned next step.
