"""Canonical paths for the AHR project.

All scripts import these constants instead of hard-coding filenames, so the
folder layout can change without breaking entry points. Paths are absolute,
anchored at the repo root via this file's location, so scripts work no matter
the caller's cwd.
"""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Directories ──────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(ROOT, 'data')
RAW_DIR      = os.path.join(DATA_DIR, 'raw')
SYNTH_DIR    = os.path.join(DATA_DIR, 'synth')
MODELS_DIR   = os.path.join(ROOT, 'models')
OUTPUTS_DIR  = os.path.join(ROOT, 'outputs')
FIGURES_DIR  = os.path.join(OUTPUTS_DIR, 'figures')
LOGS_DIR     = os.path.join(OUTPUTS_DIR, 'logs')
NEW_GLYPHS_DIR = os.path.join(RAW_DIR, 'new_glyphs')
REAL_TEST_DIR  = os.path.join(RAW_DIR, 'real_test')

# ── Raw inputs ───────────────────────────────────────────────────────────────
GLYPH_BANK_NPZ  = os.path.join(RAW_DIR, 'arabic_handwritten_dataset.npz')
DICTIONARY_JSON = os.path.join(RAW_DIR, 'ar_dictionary.json')
# Raw kaikki dump is only kept around when actively building the dictionary.
# `python scripts/dictionary.py fetch` re-downloads it; keep this constant so the
# fetcher knows where to write.
KAIKKI_JSONL    = os.path.join(RAW_DIR, 'kaikki_arabic.jsonl')

# ── Synthetic corpora ────────────────────────────────────────────────────────
# Canonical corpus = the dictionary-driven one with full ar/en/pos metadata.
# Older small corpora used during bring-up have been removed.
SYNTH_CORPUS          = os.path.join(SYNTH_DIR, 'synth_corpus_dict.npz')
SYNTH_CORPUS_ENRICHED = SYNTH_CORPUS                 # legacy alias
SYNTH_CORPUS_5K       = os.path.join(SYNTH_DIR, 'synth_corpus_5k.npz')
SYNTH_CORPUS_DICT     = SYNTH_CORPUS

# ── Models ───────────────────────────────────────────────────────────────────
EXPERT_MODEL = os.path.join(MODELS_DIR, 'expert_best.keras')
CRNN_MODEL   = os.path.join(MODELS_DIR, 'crnn_arabic_5k.keras')

# ── Output figures ───────────────────────────────────────────────────────────
SAMPLES_GRID_PNG = os.path.join(FIGURES_DIR, 'samples_grid.png')


def ensure_dir(path: str) -> str:
    """Create the parent directory of `path` if missing; return `path`."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    return path
