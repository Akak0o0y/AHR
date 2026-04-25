"""Build AHR_DHAMER_fixed.ipynb in the exact order requested.

1.  Imports & Setup
2.  Data — AHCD (Kaggle)
3.  Foundation Letters (LETTERS.png)
4.  Start Annotating Your Dataset
5.  Preparing the Arabic Handwritten Dataset
6.  Labeling Arabic Letter Forms
7.  Train on the Labeled Letter Data (foundation)
8.  Train on the Kaggle Dataset (AHCD)
9.  Word-Level 200k Images
10. Advanced Detection with RT-DETR
"""
import json

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS & SETUP
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "# Arabic Handwritten Recognition (AHR) Pipeline\n"
    "Full pipeline: imports → data → foundation letters → labeling → training → 200k words → RT-DETR."
))

cells.append(md("## 1. Imports & Setup"))

cells.append(code("""\
import os, sys, json, collections, subprocess
import cv2
import numpy as np
import pandas as pd
import pathlib
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for nbconvert
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import kagglehub

print(f"TensorFlow {tf.__version__}  |  Python {sys.version.split()[0]}")
print("GPU available:", bool(tf.config.list_physical_devices('GPU')))\
"""))

cells.append(code("""\
# ── Arabic alphabet (28 letters, AHCD order) + positional forms ───────────────
ARABIC_ALPHABET = [
    'أ','ب','ت','ث','ج','ح','خ','د','ذ','ر','ز','س','ش','ص',
    'ض','ط','ظ','ع','غ','ف','ق','ك','ل','م','ن','هـ','و','ي'
]
FORMS         = ['Isolated', 'Initial', 'Medial', 'Final']
LETTERS_IMAGE = 'data/LETTERS.png'
print(f"Alphabet: {len(ARABIC_ALPHABET)} letters  |  Forms: {FORMS}")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA — AHCD (KAGGLE)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 2. Data — AHCD (Kaggle)\n"
    "Download via `kagglehub`, load train/test CSVs, fix orientation."
))

cells.append(code("""\
path = kagglehub.dataset_download("mloey1/ahcd1")
train_images_csv = os.path.join(path, 'csvTrainImages 13440x1024.csv')
train_labels_csv = os.path.join(path, 'csvTrainLabel 13440x1.csv')
test_images_csv  = os.path.join(path, 'csvTestImages 3360x1024.csv')
test_labels_csv  = os.path.join(path, 'csvTestLabel 3360x1.csv')

X_ahcd_train = pd.read_csv(train_images_csv, header=None).values.reshape(-1,32,32,1).astype('float32')/255.0
y_ahcd_train = pd.read_csv(train_labels_csv, header=None).values.flatten() - 1
X_ahcd_test  = pd.read_csv(test_images_csv,  header=None).values.reshape(-1,32,32,1).astype('float32')/255.0
y_ahcd_test  = pd.read_csv(test_labels_csv,  header=None).values.flatten() - 1

# Fix orientation (AHCD CSVs are stored transposed + mirrored)
X_ahcd_train = np.flip(np.rot90(X_ahcd_train, k=-1, axes=(1,2)), axis=2)
X_ahcd_test  = np.flip(np.rot90(X_ahcd_test,  k=-1, axes=(1,2)), axis=2)

print(f"AHCD  train: {X_ahcd_train.shape}  labels: {y_ahcd_train.shape}")
print(f"AHCD  test : {X_ahcd_test.shape }  labels: {y_ahcd_test.shape }")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 3. FOUNDATION LETTERS
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 3. Foundation Letters — LETTERS.png\n"
    "Detect all letter regions from the annotated chart using OpenCV contour detection."
))

cells.append(code("""\
def extract_letter_regions(image_path, min_area=10, kernel_size=5, line_tolerance=60):
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated   = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    boxes = sorted(boxes, key=lambda b: (b[1] // line_tolerance, -b[0]))  # top→bottom, right→left

    dataset = []
    for i, (x, y, w, h) in enumerate(boxes):
        roi = cv2.resize(gray[y:y+h, x:x+w], (32,32), interpolation=cv2.INTER_AREA)
        dataset.append({'id': i+1, 'image': roi.tolist(), 'label': None})
    return dataset, boxes

dataset, final_boxes = extract_letter_regions(LETTERS_IMAGE)
print(f"Detected {len(dataset)} regions in {LETTERS_IMAGE}")\
"""))

cells.append(code("""\
# Manual boxes missed by contour detection (appended to dataset + final_boxes)
MANUAL_BOXES = [
    {"id": "3Z", "x": 425.5, "y": 575.5, "width": 33, "height": 21},
    {"id": "3b", "x": 296.5, "y": 575.5, "width": 33, "height": 21},
    {"id": "3d", "x": 199.5, "y": 574,   "width": 21, "height": 22},
    {"id": "3f", "x": 70.5,  "y": 576,   "width": 23, "height": 18},
]

_gray_letters = cv2.cvtColor(cv2.imread(LETTERS_IMAGE), cv2.COLOR_BGR2GRAY)
for b in MANUAL_BOXES:
    x, y, w, h = int(round(b['x'])), int(round(b['y'])), int(b['width']), int(b['height'])
    roi = cv2.resize(_gray_letters[y:y+h, x:x+w], (32,32), interpolation=cv2.INTER_AREA)
    if not any(d['id'] == b['id'] for d in dataset):
        dataset.append({'id': b['id'], 'image': roi.tolist(), 'label': None})
        final_boxes.append((x, y, w, h))

# Ensure refined_labels has entries for the new IDs (safe re-run)
if 'refined_labels' in globals():
    for b in MANUAL_BOXES:
        refined_labels.setdefault(b['id'], {'char': '', 'form': 'Isolated'})

print(f"After manual boxes: {len(dataset)} regions total")\
"""))

cells.append(code("""\
def visualize_detections(image_path, boxes, figsize=(20,25)):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    for i, (x, y, w, h) in enumerate(boxes):
        ax.add_patch(patches.Rectangle((x,y), w, h, fill=False, edgecolor='red', lw=1))
        ax.text(x, y-5, str(i+1), color='blue', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, pad=1, edgecolor='none'))
    ax.axis('off')
    ax.set_title(f'{len(boxes)} detections', fontsize=14)
    plt.tight_layout()
    plt.savefig('detections_map.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("Saved detections_map.png")

visualize_detections(LETTERS_IMAGE, final_boxes)\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 4. START ANNOTATING YOUR DATASET
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 4. Start Annotating Your Dataset\n"
    "Interactive labeler for the letter-image regions. Each region gets a **char** + **form** "
    "(Isolated / Initial / Medial / Final / Noise). Run in a **live Jupyter session**."
))

cells.append(code("""\
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    if 'refined_labels' not in globals():
        refined_labels = {item['id']: {'char': '', 'form': 'Isolated'} for item in dataset}

    def launch_batch_labeler(dataset):
        output   = widgets.Output()
        progress = widgets.IntProgress(value=0, min=0, max=len(dataset),
                                       description='Progress:', bar_style='info')
        page_size    = 10
        current_page = [0]

        def update_progress():
            progress.value = sum(1 for v in refined_labels.values() if v.get('char') != '')

        def save_batch(controls):
            if not controls: return
            for bid, (char_w, form_w) in controls.items():
                refined_labels[bid] = {'char': char_w.value.strip(), 'form': form_w.value}
            update_progress()

        def show_page(page_idx):
            start = page_idx * page_size
            end   = min(start + page_size, len(dataset))
            with output:
                clear_output(wait=True)
                grid_items, current_controls = [], {}
                for i in range(start, end):
                    item = dataset[i]; bid = item['id']
                    img_out = widgets.Output()
                    with img_out:
                        plt.figure(figsize=(1.2,1.2))
                        plt.imshow(np.array(item['image']), cmap='gray')
                        plt.axis('off'); plt.title(f"ID {bid}", fontsize=9); plt.show()
                    val     = refined_labels.get(bid, {'char':'', 'form':'Isolated'})
                    char_in = widgets.Text(value=val['char'], placeholder='Char',
                                           layout=widgets.Layout(width='70px'))
                    form_in = widgets.Dropdown(options=['Isolated','Initial','Medial','Final','Noise'],
                                               value=val['form'], layout=widgets.Layout(width='90px'))
                    current_controls[bid] = (char_in, form_in)
                    grid_items.append(widgets.VBox([img_out, char_in, form_in],
                                                    layout=widgets.Layout(border='1px solid #ccc',
                                                                          padding='5px')))
                display(widgets.HBox(grid_items, layout=widgets.Layout(flex_flow='row wrap')))
            return current_controls

        active_controls = show_page(0)

        def on_prev(b):
            save_batch(active_controls); current_page[0] = max(0, current_page[0]-1)
            active_controls.update(show_page(current_page[0]))
        def on_next(b):
            save_batch(active_controls)
            if (current_page[0]+1)*page_size < len(dataset):
                current_page[0] += 1; active_controls.update(show_page(current_page[0]))
        def on_submit(b):
            save_batch(active_controls); export_labeled_dataset(dataset, refined_labels)

        prev_btn   = widgets.Button(description="Prev Page")
        next_btn   = widgets.Button(description="Next Page")
        submit_btn = widgets.Button(description="Submit & Export", button_style='danger')
        prev_btn.on_click(on_prev); next_btn.on_click(on_next); submit_btn.on_click(on_submit)
        display(progress, output, widgets.HBox([prev_btn, next_btn, submit_btn]))

    # Uncomment to launch in a live Jupyter session:
    # launch_batch_labeler(dataset)
    print("Labeler defined. Run launch_batch_labeler(dataset) in live Jupyter.")

except Exception as e:
    print(f"Widget setup skipped: {e}")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 5. PREPARING THE ARABIC HANDWRITTEN DATASET
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 5. Preparing the Arabic Handwritten Dataset\n"
    "Export labeled regions to `.npz` for training."
))

cells.append(code("""\
def export_labeled_dataset(dataset, labels_dict, out_path='arabic_handwritten_dataset.npz'):
    images, chars, forms = [], [], []
    for item in dataset:
        info = labels_dict.get(item['id'], {})
        if info.get('char','') and info.get('form') != 'Noise':
            images.append(np.array(item['image'], dtype=np.uint8))
            chars.append(info['char'])
            forms.append(info['form'])
    if not images:
        print("No labeled samples yet — run the labeler first."); return False
    np.savez_compressed(out_path,
                        images=np.array(images),
                        char_labels=np.array(chars),
                        form_labels=np.array(forms))
    print(f"Exported {len(images)} samples → {out_path}")
    return True

# Run after labeling:
# export_labeled_dataset(dataset, refined_labels)\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 6. LABELING ARABIC LETTER FORMS (augment the small labeled set 10×)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 6. Labeling Arabic Letter Forms\n"
    "Augment the small labeled letter-form dataset 10× using rotation, shift, zoom — "
    "giving the foundation model enough variety to learn from."
))

cells.append(code("""\
def augment_labeled_dataset(src='arabic_handwritten_dataset.npz',
                             dst='arabic_augmented_dataset.npz', multiplier=10):
    if not os.path.exists(src):
        print(f"{src} not found — run export_labeled_dataset first."); return
    data  = np.load(src, allow_pickle=True)
    X     = data['images'].reshape(-1,32,32,1).astype('float32')/255.0
    chars = data['char_labels']; forms = data['form_labels']
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                 height_shift_range=0.1, zoom_range=0.1, fill_mode='nearest')
    imgs_out, chars_out, forms_out = [], [], []
    for i in range(len(X)):
        imgs_out.append(X[i]); chars_out.append(chars[i]); forms_out.append(forms[i])
        for _ in range(multiplier - 1):
            batch = next(datagen.flow(X[i:i+1], batch_size=1))
            imgs_out.append(batch[0]); chars_out.append(chars[i]); forms_out.append(forms[i])
    np.savez_compressed(dst, images=np.array(imgs_out),
                        char_labels=np.array(chars_out), form_labels=np.array(forms_out))
    print(f"Augmented {len(X)} → {len(imgs_out)} samples saved to {dst}")

# Run after exporting:
# augment_labeled_dataset()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAIN ON THE LABELED LETTER DATA (foundation)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 7. Train the Foundation Model on the Labeled Letter Data\n"
    "First training pass: teach the model the positional-form classes from your annotated LETTERS.png chart."
))

cells.append(code("""\
def train_foundation_on_letters(src='arabic_augmented_dataset.npz'):
    if not os.path.exists(src):
        print(f"{src} not found — skip this cell until you've labeled + augmented.")
        return None, None, None

    data    = np.load(src, allow_pickle=True)
    X       = data['images'].reshape(-1,32,32,1).astype('float32')
    # Class name = baseLetter_form (e.g., 'ب_Initial')
    class_names = np.array([f"{c}_{f}" for c, f in zip(data['char_labels'], data['form_labels'])])

    le          = LabelEncoder()
    y           = le.fit_transform(class_names)
    num_classes = len(le.classes_)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if num_classes > 1 else None)

    m = models.Sequential([
        layers.Input(shape=(32,32,1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    h  = m.fit(X_tr, y_tr, epochs=50, batch_size=32,
               validation_data=(X_va, y_va), callbacks=[es], verbose=1)
    m.save('foundation_letter_model.keras')
    print(f"Foundation model saved. Classes ({num_classes}): {list(le.classes_)[:10]}...")
    return m, le, h

model_foundation, le_foundation, history_foundation = train_foundation_on_letters()\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 8. TRAIN ON THE KAGGLE DATASET (AHCD)
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 8. Train on the Kaggle Dataset (AHCD)\n"
    "Second training pass: large-scale pre-training on 13,440 AHCD samples (28 classes, isolated letters). "
    "This produces the **expert model** used for word-level prediction and 200k auto-labeling."
))

cells.append(code("""\
def build_arabic_expert(num_classes=28):
    return models.Sequential([
        layers.Input(shape=(32,32,1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(), layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'), layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])

model_expert = build_arabic_expert()
model_expert.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_expert.summary()\
"""))

cells.append(code("""\
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ckpt       = callbacks.ModelCheckpoint('expert_best.keras', save_best_only=True, monitor='val_accuracy')

history_expert = model_expert.fit(
    X_ahcd_train, y_ahcd_train,
    epochs=25, batch_size=64,
    validation_data=(X_ahcd_test, y_ahcd_test),
    callbacks=[early_stop, ckpt],
)
print("Expert training complete. Best model saved to expert_best.keras")\
"""))

cells.append(code("""\
# ── Evaluate the expert on AHCD test set ──────────────────────────────────────
loss, acc = model_expert.evaluate(X_ahcd_test, y_ahcd_test, verbose=0)
print(f"Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

y_pred_expert = np.argmax(model_expert.predict(X_ahcd_test, verbose=0), axis=1)
print(classification_report(y_ahcd_test, y_pred_expert,
                            target_names=ARABIC_ALPHABET, zero_division=0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_expert.history['accuracy'], label='train')
ax1.plot(history_expert.history['val_accuracy'], label='val')
ax1.set_title('Expert Accuracy'); ax1.legend()
ax2.plot(history_expert.history['loss'], label='train')
ax2.plot(history_expert.history['val_loss'], label='val')
ax2.set_title('Expert Loss'); ax2.legend()
plt.tight_layout()
plt.savefig('expert_training_curve.png', dpi=100)
plt.close()
print("Saved expert_training_curve.png")

cm = confusion_matrix(y_ahcd_test, y_pred_expert)
fig, ax = plt.subplots(figsize=(14,12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=ARABIC_ALPHABET, yticklabels=ARABIC_ALPHABET, ax=ax)
ax.set_title('Confusion Matrix — Expert Model (AHCD test)')
ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('expert_confusion_matrix.png', dpi=100)
plt.close()
print("Saved expert_confusion_matrix.png")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 9. WORD-LEVEL 200K IMAGES
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 9. Word-Level 200k Images\n"
    "Use the expert model to predict characters per word (RTL segmentation) and "
    "auto-label a 200k-image corpus with a confidence threshold."
))

cells.append(code("""\
def predict_word(image_path, boxes, model=None, alphabet=ARABIC_ALPHABET):
    \"\"\"RTL segmentation → predicted Arabic word.\"\"\"
    if model is None: model = model_expert
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    rtl_boxes = sorted(boxes, key=lambda b: -b[0])
    batch = []
    for (x, y, w, h) in rtl_boxes:
        roi = img[y:y+h, x:x+w]
        if roi.size == 0: continue
        batch.append(cv2.resize(roi, (32,32)).reshape(32,32,1).astype('float32')/255.0)
    if not batch: return ''
    preds = model.predict(np.array(batch), verbose=0)
    return ''.join(alphabet[i] for i in np.argmax(preds, axis=1))\
"""))

cells.append(code("""\
def batch_auto_label(root_folder, model=None, confidence_threshold=0.95,
                     out_csv='large_scale_auto_labels.csv'):
    if model is None: model = model_expert
    exts = ['*.png','*.jpg','*.jpeg','*.PNG','*.JPG','*.JPEG']
    all_files = []
    for ext in exts:
        all_files.extend(pathlib.Path(root_folder).rglob(ext))
    print(f"Found {len(all_files)} images in {root_folder}")

    results = []
    for i in tqdm(range(0, len(all_files), 128)):
        batch_files = all_files[i:i+128]
        imgs, paths = [], []
        for p in batch_files:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(cv2.resize(img,(32,32)).reshape(32,32,1).astype('float32')/255.0)
                paths.append(str(p))
        if not imgs: continue
        preds = model.predict(np.array(imgs), verbose=0)
        for idx, pred in enumerate(preds):
            conf = float(np.max(pred))
            if conf >= confidence_threshold:
                results.append({
                    'path': paths[idx],
                    'folder': os.path.basename(os.path.dirname(paths[idx])),
                    'predicted_char': ARABIC_ALPHABET[int(np.argmax(pred))],
                    'confidence': conf,
                })
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"Auto-labeled {len(results)}/{len(all_files)} images (conf>={confidence_threshold})")
    print(f"Results saved to {out_csv}")
    return df

# ── Run auto-labeling on your 200k dataset (set your path) ────────────────────
# unlabeled_dir = r'C:/path/to/your/200k/word/images'
# labels_df = batch_auto_label(unlabeled_dir)
# labels_df.head()
print("Set unlabeled_dir and uncomment the lines above to run 200k auto-labeling.")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# 10. ADVANCED DETECTION WITH RT-DETR
# ══════════════════════════════════════════════════════════════════════════════
cells.append(md(
    "## 10. 🎯 Advanced Detection with RT-DETR\n"
    "Stage 3 (future): detect full-page handwritten word/line regions with RT-DETR, "
    "then pass each crop to the expert model for recognition."
))

cells.append(code("""\
# Install once from terminal:  pip install ultralytics
try:
    from ultralytics import RTDETR
    print("Ultralytics ready.")

    def detect_with_rtdetr(image_path, model_path='rtdetr-l.pt', conf=0.25):
        det_model = RTDETR(model_path)
        results   = det_model(image_path, conf=conf)
        boxes_out = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                boxes_out.append((x1, y1, x2-x1, y2-y1))
        return boxes_out

    # Example end-to-end pipeline (needs a real document image):
    # boxes = detect_with_rtdetr('path/to/page.png')
    # word  = predict_word('path/to/page.png', boxes)
    print("RT-DETR stub ready. Fine-tune on annotated document images for Stage 3.")

except Exception as e:
    print(f"Ultralytics not installed yet: {e}")\
"""))

# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE & WRITE
# ══════════════════════════════════════════════════════════════════════════════
# Assign stable cell IDs
for i, c in enumerate(cells):
    c['id'] = f"cell-{i:02d}"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = 'AHR_DHAMER_fixed.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {len(cells)} cells to {out}")
