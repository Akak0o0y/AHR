"""Evaluate the trained CRNN on the full synth corpus.

Reports:
  - Per-sample accuracy (exact word match)
  - Character Error Rate (CER) — Levenshtein distance / ground-truth length
  - A handful of mispredictions for qualitative inspection
"""
from __future__ import annotations
import sys, io
import numpy as np
import tensorflow as tf
from solution import (
    build_crnn, ctc_loss_fn, decode_prediction, IDX_TO_CHAR,
)
from train import load_corpus
import paths


def levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    corpus = paths.SYNTH_CORPUS_ENRICHED
    model_path = paths.CRNN_MODEL

    X, y, _ = load_corpus(corpus)
    print(f"Loading {model_path} …")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={'ctc_loss_fn': ctc_loss_fn})

    print(f"Predicting on {len(X)} samples …")
    preds = model.predict(X, batch_size=64, verbose=1)

    n = len(X)
    n_exact = 0
    total_chars = 0
    total_edits = 0
    misses: list[tuple[str, str]] = []
    for i in range(n):
        true_ids = y[i][y[i] != -1]
        true = ''.join(IDX_TO_CHAR[int(t)] for t in true_ids)
        pred = decode_prediction(preds[i])
        total_chars += len(true)
        d = levenshtein(true, pred)
        total_edits += d
        if pred == true:
            n_exact += 1
        elif len(misses) < 15:
            misses.append((true, pred))

    acc = n_exact / n
    cer = total_edits / max(total_chars, 1)
    print(f"\n=== RESULTS on {corpus} ({n} samples, {len(set([''.join(IDX_TO_CHAR[int(t)] for t in y[i][y[i] != -1]) for i in range(n)]))} unique words) ===")
    print(f"Exact-match accuracy: {n_exact}/{n} = {acc*100:.2f}%")
    print(f"Character Error Rate (CER): {cer*100:.2f}%")
    if misses:
        print(f"\nFirst {len(misses)} mispredictions:")
        for true, pred in misses:
            print(f"  true={true!r:15s}  pred={pred!r}")


if __name__ == '__main__':
    main()
