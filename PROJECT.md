# Arabic Handwritten Recognition (AHR)
## Vision, Theory, Architecture, and Roadmap

> A CTC-aligned handwriting recognizer for Arabic, trained on synthetic words
> composed from a bank of real handwritten letters, bootstrapped by a
> letter-form classifier and enriched with a Wiktionary-sourced dictionary
> so downstream models can learn image → letters + word + meaning.

---

## 1. Vision

The long-term goal is a three-stage pipeline that takes a raw image of a
handwritten Arabic document and returns structured, grounded output:

```
   Document image
        ↓
   Stage 3: Word detection          ← RT-DETR (future)
        ↓  (word crops)
   Stage 2: Word OCR                ← CRNN + CTC  (this repo)
        ↓  (letter sequences)
   Stage 1: Letter-form classifier  ← already trained (expert_best.keras)
        ↓  (optional re-scoring at letter level)
   Enriched output: letters + word + Arabic/English definition
```

This document scopes **Stage 2 (word OCR)**, which is the central contribution
of this repo. Stages 1 and 3 are referenced for context but not implemented
here. The eventual research paper should frame the contribution as:

> *A segmentation-free Arabic handwriting recognizer whose training data is
> synthesized from a small handwritten letter bank using positional-form-aware,
> pixel-level composition that respects Arabic cursive ligature rules; enriched
> with a dictionary layer so a single image can be labeled with both the word
> and its meaning.*

---

## 2. Scope

**In scope**
- Word-level recognition of isolated handwritten Arabic words (not lines, not
  paragraphs).
- Training on synthetic data composed from real handwritten letter samples
  (not font renderings).
- Dictionary enrichment so each training sample has `(image, word,
  ar_definition, en_definition)`.

**Out of scope (but planned)**
- Line- or page-level recognition (Stage 3 above).
- Multi-writer style diversity beyond the existing letter bank — current bank
  is effectively a single source of handwriting style.
- Arabic diacritics (tashkeel) in output — normalized away during training.
- Hamza-variant *surface forms* — normalized to a canonical 28-letter set; the
  model will never output `أ إ آ ى ة ؤ ئ ء`.

---

## 3. Theory: Why every design choice is what it is

This section is the one the paper should lean on hardest.

### 3.1 Why not segmentation-then-recognition

The naive pipeline — segment a word into letters, classify each letter, stitch
back together — collapses on connected Arabic cursive.

The problem: Arabic letters connect along a baseline *kashida* (ـ), so the
boundary between two letters is often a **continuous stroke**, not a gap.
Vertical-projection segmentation tries to find "necks" in the ink profile,
but these necks do not correspond to linguistic letter boundaries. In our
original experiments the expert letter-form classifier would collapse every
word into `"ررررر"` — the model would rather emit a high-confidence prediction
for the most-frequent shape than produce calibrated "I don't know" outputs
at bogus boundaries.

**Corollary rule:** a word recognizer must handle the *alignment* problem
between image time-steps and output characters. This is what CTC does.

### 3.2 Why CRNN + CTC

CTC (Connectionist Temporal Classification) learns the alignment between an
input sequence of length T (our feature-map columns after the CNN) and an
output sequence of length L ≤ T (our letter indices) without requiring
per-timestep labels. Training does not need segmentation.

The math in one line:
```
   P(label | image) = Σ over all alignments a that map to label: Π P(a_t | image, t)
```

At inference we use greedy decode: argmax per time-step → collapse consecutive
duplicates → drop blanks. For inference improvements, we can later swap to
beam search with a lexicon or language model.

Input constraints the architecture must respect:
- `T` (time-steps after the CNN) must be ≥ `L` (letter count). Our CNN keeps
  width stride at `DOWNSAMPLE=4`, so for `IMG_W=128` → `T=32`, comfortably
  larger than the max label length of any renderable 10-letter word.
- The blank token is **class index 28** (right after the 28-letter alphabet).

### 3.3 Why synthetic training data (and this flavor of it)

**Distribution matching beats data volume.**

Two tempting-but-wrong options:
1. Train on printed Arabic text with a font. → The model learns a font domain
   and never transfers to messy handwriting.
2. Self-supervised training on the 200k unlabeled real handwriting corpus
   (MUKF4A). → Possible as Phase 2, but there is no labeled signal to bootstrap
   from; we need a seed supervised model first.

Instead we synthesize `(image, label)` pairs where the image is assembled from
**real handwritten letter samples** we already labeled for the letter-form
classifier. This keeps the training distribution stylistically close to real
handwriting: same ink, same noise, same stroke wobble.

**Why this is not "mocking" or a hack:** The synthetic step is a *supervised
data generator*, not a stand-in for real data. Every pixel of ink in a synth
word image originated from a real handwritten letter sample. The synthesis
only determines which letters go next to which, and how they connect.

### 3.4 Orthographic vs. semantic knowledge

A core conceptual distinction the user raised:

- **Orthographic knowledge:** which letter sequences are valid Arabic words.
  Models pick this up from the dictionary-derived word list because we only
  train on real words. That alone improves accuracy on real handwriting
  because ambiguous letters resolve to the more-likely-valid spelling.
- **Semantic knowledge:** what a word *means*. The recognizer never learns
  this. That is why we store `ar_definitions` and `en_definitions` alongside
  labels — so a downstream model can be trained on `image → meaning` or
  `image → translation`, with the OCR output as an intermediate target.

The paper should be explicit: Stage 2 produces letter sequences. Meaning is
carried as metadata, not learned from images.

### 3.5 Why bidirectional LSTM (not just forward)

Arabic letters are disambiguated not just by the stroke shape but by **dots
above and below** that may be drawn at different timing relative to the main
stroke. Because the dots appear above/below at the same time step as the main
letter, the temporal features at time `t` must see context from both sides.
A forward-only LSTM would underperform on dot-dependent pairs like
(ج، ح، خ) and (ب، ت، ث).

We use two stacked BiLSTM(128) layers with dropout 0.25.

### 3.6 Arabic normalization: why the 28-letter canonical alphabet

Labels must be consistent. The Arabic writing system has several hamza and
maddah variants (أ إ آ ؤ ئ ء ة ى) that are *orthographic surface forms* of
a smaller set of canonical letters. For OCR we normalize:

```
   أ إ آ → ا       alif-hamza variants → bare alif
   ى → ي           alif maqsura → ya
   ة → ه           taa marbuta → ha
   ؤ → و           waw-hamza → waw
   ئ → ي           ya-hamza → ya
   tatweel, hamza-on-line, diacritics → dropped
```

This is the standard NLP-canonical set used by ElixirFM, Farasa, CAMeL Tools.
**Rule:** anything reading or writing labels must run `word.translate(NORMALIZE)`
first. Skipping this causes silent letter-dropping bugs (we already hit one
where `'رمادي'` became `'رمدي'` because bare `ا` was missing from the
alphabet).

### 3.7 Positional forms and PAW boundaries

Arabic letters have up to four positional forms: **Isolated, Initial, Medial,
Final**. Six letters are **non-connectors** (ا د ذ ر ز و) plus ة — they do
not connect to the *following* letter. After a non-connector, the run breaks
and the next letter starts fresh as Initial (or Isolated if it's the last
letter of the word). Each connected run is a PAW — *Piece of Arabic Word*.

`pick_form(word, i)` in [glyph_bank.py](glyph_bank.py) implements these rules:

```
   single-letter word           → Isolated
   current is non-connector:
     previous connects to it    → Final
     else                       → Isolated
   current is a connector:
     previous did not connect   → Initial   (or Isolated if it's also last)
     is last letter of word     → Final
     else                       → Medial
```

**Why the model benefits:** each form has a different glyph. Training on the
right per-position glyph teaches the CRNN to recognize letters in context,
matching how the real-world test images look.

### 3.8 Pixel-aware composition: baseline alignment and matched connectors

Naive glyph-strip composition (paste next letter to the right of the previous
one) leaves visible seams — letters at different heights, broken connectors,
mismatched stroke widths. These seams become a feature the CRNN learns to
exploit, which does not generalize to real handwriting.

[word_composer.py](word_composer.py) does pixel-level matching:
1. For each glyph, find the **baseline band** — the 3-px-tall horizontal
   region where inter-letter connectors run.
2. Compute `_edge_profile(glyph, side, baseline_y)`: on the connecting side
   of the glyph, find the vertical center-of-mass of ink inside the baseline
   band, plus the thickness of that ink run. This is the glyph's "exit point"
   (previous letter, left side) or "entry point" (current letter, right side,
   since Arabic reads right-to-left).
3. Y-shift the incoming glyph so the two y_centers align.
4. Draw a short cv2.line connector of *matched thickness* between the two
   points, overlapping by `CONNECT_OVERLAP=3` px.

This is the "the model draws by editing pixels" step the user asked for: we
are not stamping TTF glyphs, we are continuously deforming real ink so the
composition looks like one continuous hand-drawn stroke.

### 3.9 RTL composition without pixel-mirroring

Arabic is read right-to-left, so in the rendered image the first letter of the
word sits on the **right** of the canvas, the last on the left. Critically,
**individual letter pixels must NOT be mirrored** — only the layout direction
changes. Our implementation handles this by running the composer cursor from
the right edge of the canvas to the left, pasting each glyph in its original
orientation.

At training time, `train.py` applies one global `X[:, :, ::-1, :]` horizontal
flip to the whole batch so that image columns (time-steps) align with label
character order. This keeps the CTC loss computation LTR, which is the only
direction Keras' `ctc_batch_cost` supports. At real-world inference on RTL
handwriting, the same flip is applied before feeding the image in.

### 3.10 Glyph fallback cascade

The letter bank is incomplete — not every `(letter, form)` pair has a sample.
`GlyphBank._resolve` tries, in order:
1. `(letter, form)` — exact match.
2. `(letter, Isolated)`, then Final/Initial/Medial — form-level fallback.
3. For each visual-equivalent in `EQUIVALENTS[letter]`, repeat the form cascade.
   (Example: if bare `ا` is missing we fall through to `أ`, which visually is
   alif with an extra hamza the classifier will learn to ignore.)

The fallback is the reason we can render the full 150-word starter list with
a bank that only has ~100 `(letter, form)` samples.

### 3.11 Augmentation: per-call, not per-sample

Each `GlyphBank.sample(...)` call runs the full augmentation stack fresh:
elastic distortion, stroke dilate/erode, rotation ±3°, translation ±1 px,
brightness noise ±10. That means the same underlying letter image is never
pasted twice looking identical — even when only 1-2 raw samples exist per
`(letter, form)`. This is what lets 100 raw letter samples produce 750+
visually-distinct word images without overfitting to pixel-exact memorization.

### 3.12 Dictionary enrichment and downstream training

`ar_dictionary.json` (54,846 entries, sourced from kaikki.org's English
Wiktionary Arabic dump via `dictionary.py fetch`) maps each renderable word
to its English gloss and part-of-speech. `synth_corpus.py` writes these
alongside images and labels in the `.npz`:

```
   synth_corpus_enriched.npz:
     images:          (N,)   object, uint8 grayscale, variable width
     labels:          (N,)   str
     ar_definitions:  (N,)   str     (empty on current dump; see 3.13)
     en_definitions:  (N,)   str
```

**Training rule:** the CRNN (`train.py`) reads only `images` and `labels`.
The definition arrays are dormant metadata for later image→meaning models.
This separation is deliberate — CTC only aligns characters; trying to learn
meaning directly from the image would require a much larger labeled corpus.

### 3.13 Known theoretical gap: single-writer source

All handwritten letter samples came from one dataset with relatively uniform
style. Augmentation partially compensates (stroke jitter, elastic), but the
style variance in the synth corpus is narrower than in real multi-writer
handwriting. This limits generalization. Mitigations on the roadmap:
1. Add a second letter-sample bank from a different dataset.
2. Style-transfer augmentation using unlabeled real writing.
3. Phase 2 pseudo-labeling on the 200k real MUKF4A corpus, which exposes
   the model to real multi-writer variance without new manual labels.

### 3.14 Known theoretical gap: Arabic-only definitions

`ar_definitions` is empty because the English Wiktionary dump only carries
English glosses for Arabic headwords. A second pass against the Arabic
Wiktionary dump (`kaikki.org/arwiktionary/...`) will fill this field.

---

## 4. Model Architecture

### 4.1 Overall shape

```
   Input (B, 32, 128, 1) grayscale, already flipped LTR for CTC
       │
       ▼
   Conv 32·3x3 + BN + ReLU + MaxPool 2x2     → (16, 64, 32)
   Conv 64·3x3 + BN + ReLU + MaxPool 2x2     → ( 8, 32, 64)
   Conv128·3x3 + BN + ReLU + MaxPool 2x1     → ( 4, 32,128)
   Conv256·3x3 + BN + ReLU + MaxPool 2x1     → ( 2, 32,256)
   Conv256·2x2 valid                         → ( 1, 31,256)
       │
       ▼
   Reshape → (T=31, 256)
       │
       ▼
   BiLSTM(128, return_sequences=True, dropout 0.25)   → (T, 256)
   BiLSTM(128, return_sequences=True, dropout 0.25)   → (T, 256)
       │
       ▼
   Dense(29, softmax)  — 28 letters + 1 blank
       │
       ▼
   Logits (B, T=31, 29)
```

### 4.2 Backbone warm-start

`build_crnn(backbone_weights='expert_best.keras')` copies matching Conv2D and
BatchNormalization weights from the pretrained letter-form classifier. This
is transfer learning in the strict sense: we keep the feature extractor that
already knows Arabic letter shapes, and only the sequence head is trained
from scratch.

### 4.3 Loss

```python
def ctc_loss_fn(y_true, y_pred):
    batch     = tf.shape(y_pred)[0]
    input_len = tf.fill([batch, 1], tf.shape(y_pred)[1])         # T
    label_len = tf.math.count_nonzero(y_true + 1, -1, keepdims=True,
                                      dtype=tf.int32)            # L (non-padding)
    return K.ctc_batch_cost(y_true, y_pred, input_len, label_len)
```

Labels are right-padded with `-1`; `label_len` counts non-padding entries.

### 4.4 Inference

Greedy decode in `decode_prediction`:
1. `argmax` along class axis → (T,) class indices.
2. Collapse consecutive duplicates.
3. Drop blank (class index 28).
4. Map remaining indices through `IDX_TO_CHAR`.

Future: beam search with lexicon rescoring (constrain to known dictionary
words; tie-break by `en_definitions` presence).

---

## 5. Data

### 5.1 Handwritten letter samples (input to glyph bank)

File: `arabic_handwritten_dataset.npz`
- `images`: (N, 32, 32, 1) uint8 or float32. Grayscale, black ink on white.
- `char_labels`: (N,) Unicode strings — e.g. `'ب'`, `'كـ'`, `'ـسـ'`.
- `form_labels`: (N,) — `'Isolated' | 'Initial' | 'Medial' | 'Final'`.

Loaded by `GlyphBank.__init__`. Currently ~100 unique `(char, form)` combos.

### 5.2 Word list

Two sources:
- `WORD_LIST_STARTER` in `synth_corpus.py` — 150 common Arabic words, bundled
  so the pipeline runs out-of-the-box without downloads.
- `--word_list ar_words.txt` — one word per line. Recommended source:
  Hunspell ar_SA (~100k) from github.com/wooorm/dictionaries.

After `filter_renderable`, words are dropped if:
- length < 2 or > 10,
- any letter has no reachable glyph in the bank,
- (optional) `--require_dict` and the word is missing from the dictionary.

### 5.3 Dictionary

File: `ar_dictionary.json`
- Format: `{"word": {"ar": str|None, "en": str|None, "pos": str|None}, ...}`
- Current size: 54,846 entries
- Source: kaikki.org English-Wiktionary Arabic dump (extractor:
  [tatuylonen/wiktextract](https://github.com/tatuylonen/wiktextract) on
  GitHub). Downloaded gzipped (~38 MB) then converted to flat JSON.

### 5.4 Synthetic corpus

File: `synth_corpus.npz` / `synth_corpus_enriched.npz`
- `images`: (N,) object — uint8 arrays of shape `(48, W)` where W varies.
- `labels`: (N,) str — the logical-order Arabic word.
- `ar_definitions`, `en_definitions`: (N,) str — aligned metadata when a
  dictionary was passed to `synth_corpus.py`.

Last generation: 750 images from the 150-word starter list (5 per word),
100% dictionary-covered.

### 5.5 Real unlabeled corpus (Phase 2 target)

~200k MUKF4A handwritten word images. Not used in Phase 1. Phase 2 plan:
run the trained CRNN on every image, keep predictions with confidence > 0.9
as pseudo-labels, retrain on `synth_corpus + pseudo_labels`. Iterate 2-3
rounds. This is noisy-student / self-training in the standard sense.

---

## 6. Files — detailed responsibilities

### [solution.py](solution.py)
Defines the model and the core label contract.
- `ARABIC_ALPHABET` — the 28 canonical letters.
- `NORMALIZE` — `str.maketrans` translation table applied before encoding.
- `CHAR_TO_IDX`, `IDX_TO_CHAR`, `NUM_CLASSES`, `BLANK_IDX`.
- `IMG_H=32`, `IMG_W=128`, `DOWNSAMPLE=4` → `T = IMG_W // DOWNSAMPLE`.
- `encode_label(word)` — normalize then map to indices.
- `decode_prediction(logits)` — greedy CTC decode.
- `build_crnn(backbone_weights=...)` — the 5-Conv + 2-BiLSTM + Dense stack.
- `ctc_loss_fn(y_true, y_pred)`.
- `train(...)` — alternate entry point (most training is run via `train.py`).
- `predict_word(model, img)` — inference helper.

### [glyph_bank.py](glyph_bank.py)
The glyph lookup layer.
- `NON_CONNECTORS` — the 6 letters + ة that break PAW runs.
- `EQUIVALENTS` — visual-equivalent fallback chains for missing glyphs.
- `augment(img, rng)` — elastic + stroke jitter + rotate + translate + noise.
- `GlyphBank` class: loads the .npz into `{(char, form): [samples]}`,
  resolves fallbacks, returns augmented samples.
- `pick_form(word, i)` — PAW-aware positional form picker.

### [word_composer.py](word_composer.py)
Turns a word string into a handwritten word image.
- `_edge_profile(glyph, side, baseline_y)` — baseline-band ink analysis.
- `_draw_connector(canvas, p1, p2, thickness)` — cv2.line between exit/entry.
- `compose_word(word, bank)` — RTL layout, per-letter Y-shift to baseline,
  thickness-matched connector between connecting letters, no connection
  across non-connectors.
- Constants: `CANVAS_H=48`, `CONNECT_OVERLAP=3`, `EDGE_BAND=3`,
  `INK_THRESHOLD=128`.

### [synth_corpus.py](synth_corpus.py)
Batch-generates the .npz.
- `load_word_list(path)` — file or starter fallback.
- `filter_renderable(words, bank, dictionary=None, require_dict=False)`.
- `generate_corpus(words, bank, samples_per_word, out_path, dictionary=None)`
  — N synth images per word, saves object-dtype image array alongside
  labels and optional definitions.
- CLI flags: `--word_list`, `--glyph_bank`, `--samples_per_word`,
  `--dictionary`, `--require_dict`, `--out`, `--seed`, `--verbose`.

### [dictionary.py](dictionary.py)
Arabic dictionary I/O.
- `ArabicDictionary(path)` — loads flat JSON, normalization-aware lookup.
- `build_from_kaikki(jsonl_path, out_json)` — converts a kaikki.org JSONL
  dump to flat JSON. Majority-script classifier routes each gloss to `ar`
  or `en` so English glosses with occasional Arabic references don't get
  misclassified.
- `download_kaikki(out_jsonl, url)` — streams the gzipped dump and
  decompresses on the fly; falls back to plain .jsonl if a non-gz URL
  is passed.
- `fetch(...)` — download + convert in one step.
- CLI subcommands: `fetch`, `download`, `from_kaikki`, `test`.

### [train.py](train.py)
CRNN training entry point.
- `resize_keep_aspect(img)` — scale to IMG_H, pad right with 255 to IMG_W.
- `load_corpus(path)` — resize all images, encode all labels, pad labels
  with `-1` to uniform length, apply the LTR flip for CTC.
- `peek(model, X, y)` — greedy-decode a few samples for qualitative check.
- `main()` — builds model, compiles with Adam(1e-3), runs 50 epochs with
  EarlyStopping(patience=8), ReduceLROnPlateau(patience=4, factor=0.5),
  ModelCheckpoint('crnn_arabic.keras', save_best_only=True).
- CLI flags: `--corpus`, `--epochs`, `--batch_size`, `--backbone`, `--out`,
  `--smoke`.

### [expert_best.keras](expert_best.keras)
Pretrained letter-form classifier from Stage 1. Used only as a backbone
weight-donor for the CRNN CNN layers.

### [arabic_handwritten_dataset.npz](arabic_handwritten_dataset.npz)
The input glyph bank. See §5.1.

### [ar_dictionary.json](ar_dictionary.json)
The Arabic dictionary. See §5.3.

### [synth_corpus.npz](synth_corpus.npz), [synth_corpus_enriched.npz](synth_corpus_enriched.npz)
Generated training data. See §5.4.

### [crnn_arabic.keras](crnn_arabic.keras)
Current trained CRNN checkpoint (train loss 0.0098, val loss 3.52 on
750-sample smoke corpus — overfit on purpose to validate the pipeline;
rerun on the full Hunspell word list for production).

---

## 7. Rules / Invariants

These are load-bearing — breaking any of them silently corrupts training.

1. **Always normalize before encoding.** `encode_label` runs
   `word.translate(NORMALIZE)` first. Any other code that converts Arabic
   text to indices must do the same.
2. **Alphabet is exactly the 28 canonical letters.** Adding hamza variants
   breaks label consistency.
3. **Non-connector rule.** `pick_form` must never return `Initial` or
   `Medial` for a letter in `NON_CONNECTORS`. Composition logic must break
   the run (no connector stroke) across these letters.
4. **No pixel-mirroring of individual glyphs.** RTL is a layout property,
   not a pixel-level one. The global flip in `train.py` is the only
   legitimate mirroring step.
5. **CTC blank is class index 28.** Anything that hard-codes a different
   index breaks decoding.
6. **Extra `.npz` keys are ignored by training.** `ar_definitions` and
   `en_definitions` are downstream metadata; never wire them into the
   CTC loss.
7. **Dictionary lookups go through `_normalize` first.** Otherwise
   `'كتاب'` and `'كِتاب'` would miss each other.
8. **Variable-width images are stored as object-dtype arrays.** Do not try
   to `np.stack` them.

---

## 8. Reproducibility / Commands

```bash
# 1. Build the dictionary (~40 MB gzipped download, 55k entries)
python dictionary.py fetch --out ar_dictionary.json

# 2. Generate the synth corpus (starter list, 5 samples per word)
python synth_corpus.py --samples_per_word 5 \
    --dictionary ar_dictionary.json \
    --out synth_corpus_enriched.npz

# 3. Smoke-train the CRNN (1 epoch, no callbacks)
python train.py --smoke --corpus synth_corpus_enriched.npz

# 4. Full training (50 epochs, EarlyStopping+ReduceLR+Checkpoint)
python train.py --corpus synth_corpus_enriched.npz --epochs 50
```

For production:
```bash
# Replace the starter list with a real Arabic word list
curl https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/ar/index.dic \
    -o ar_words_raw.dic
# (parse out words from Hunspell format, or use any ~100k word list you prefer)
python synth_corpus.py --word_list ar_words.txt --samples_per_word 20 \
    --dictionary ar_dictionary.json --out synth_corpus_full.npz
python train.py --corpus synth_corpus_full.npz --epochs 50
```

---

## 9. Current results

On the 150-word starter list × 5 samples = 750 images:
- Train loss converged to **0.0098** after ~30 epochs.
- Val loss plateaued at **3.52** (overfitting, as expected on this tiny corpus).
- Peek (greedy decode on train set): **8 / 8 correct** —
  `اصفر، درس، اسود، بنت، قمر، علم، روح، خبز`.

**Interpretation:** the pipeline is end-to-end functional. The train/val gap
is the expected signature of memorization on a small word set. Production
quality requires either (a) a real word list of ≥10k unique words, or (b)
Phase 2 pseudo-labeling on the 200k real corpus.

---

## 10. Roadmap

### Phase 1 — current
- [x] CRNN + CTC architecture, letter-form backbone transfer.
- [x] Glyph bank with positional-form-aware sampling and augmentation.
- [x] Pixel-aware RTL composition with baseline-matched connectors.
- [x] Synth corpus generator, dictionary enrichment.
- [x] End-to-end smoke training.

### Phase 2 — production training
- [ ] Drop in Hunspell ar_SA word list (~100k words).
- [ ] Render ≥20 samples per word → ~2M synth corpus.
- [ ] Full 50-epoch training on the large corpus.
- [ ] Evaluate on a hand-held validation set of real handwritten words
      (target: CER < 10 %, WER < 30 %).

### Phase 3 — noisy-student on real data
- [ ] Run Phase 2 model on the 200k MUKF4A corpus.
- [ ] Keep predictions with greedy-decode confidence > 0.9.
- [ ] Retrain on `synth + pseudo_labeled_real`.
- [ ] Iterate 2-3 rounds. Expected large accuracy jump because the model
      finally sees multi-writer variance.

### Phase 4 — fine-tune on small real labels
- [ ] Hand-label 500-1000 real words from MUKF4A.
- [ ] Short fine-tune at lower LR on the labeled set.

### Phase 5 — inference improvements
- [ ] Beam search decoding with lexicon rescoring (constrain to dictionary).
- [ ] Per-form re-scoring using the Stage-1 letter classifier as a critic.
- [ ] Second dictionary pass against Arabic Wiktionary to fill `ar_definitions`.

### Deferred open directions (discussed, not yet prioritized)
- CNN-driven synthesis: have the letter classifier veto bad glyph choices
  before they hit the training set (quality gate), pick the best glyph
  variant (best-glyph picker), or learn the connector itself (learned
  connector head).
- Transformer-based OCR (Donut, TrOCR) once we have enough real labels.
- Stroke-aware augmentation using online stroke data (if we can acquire it).

---

## 11. Known limitations (be honest in the paper)

1. **Single-writer glyph source.** Current corpus has narrow style variance.
   Mitigated partially by augmentation, fully by Phase 3.
2. **No diacritics in output.** We normalize them away; if the downstream
   task requires diacritics, a second model is needed.
3. **No out-of-vocabulary handling.** CTC can emit any 28-letter sequence,
   but performance on spellings never seen in the synth corpus is
   untested and likely worse than in-vocab.
4. **ar_definitions is empty.** English Wiktionary dump only. To fix, add
   the Arabic Wiktionary dump as a second source.
5. **No line- or page-level segmentation.** Stage 3 is future work.
6. **Synthetic compositing, even pixel-aware, is a simplification.** Real
   handwriting has ligature shape variation that our composer does not
   reproduce (e.g. the `لا` ligature is drawn differently from separate
   `ل + ا`; we use the separate-glyph version).

---

## 12. Suggested paper structure

1. **Abstract** — segmentation-free Arabic OCR via CRNN + CTC, trained on
   pixel-aware synthesized handwritten words from a real letter bank.
2. **Related work** — segmentation-based approaches, CRNN/CTC on Latin OCR,
   Arabic-specific OCR challenges (cursive, dots, diacritics).
3. **Method** — §3 and §4 of this document, tightened.
4. **Data synthesis pipeline** — §3.7, §3.8, §3.9, §5 of this document.
5. **Dictionary enrichment** — §3.12, §5.3 of this document; note the
   downstream-model implications.
6. **Experiments** — Phase 1 smoke numbers + Phase 2 full-corpus numbers.
7. **Pseudo-labeling / noisy-student** — Phase 3 results.
8. **Ablations** — with/without letter-form backbone transfer, with/without
   pixel-aware composition, with/without augmentation, with/without
   positional-form awareness.
9. **Limitations** — §11.
10. **Conclusion & future work** — §10 Phases 4-5.
