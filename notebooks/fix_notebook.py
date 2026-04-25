import json

with open('AHR_DHAMER.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

# ── helpers ──────────────────────────────────────────────────────────────────
LETTERS_PATH = 'data/LETTERS.png'

def get_src(c):
    s = c['source']
    return ''.join(s) if isinstance(s, list) else s

def set_src(c, s):
    c['source'] = s

def fix_src(s):
    return s.replace("'/content/LETTERS.png'", f"'{LETTERS_PATH}'")\
             .replace('"/content/LETTERS.png"', f'"{LETTERS_PATH}"')

# Fix every cell's source for the LETTERS path
for c in cells:
    if c['cell_type'] == 'code':
        set_src(c, fix_src(get_src(c)))

# ── Cell 5: guard X_train_large before training ───────────────────────────────
# Find cell that calls model_arabic_expert.fit with X_train_large (the early one, index ~5)
for i, c in enumerate(cells):
    if c['cell_type'] == 'code' and 'model_arabic_expert.fit' in c['source'] and 'if' not in c['source'] and i < 10:
        c['source'] = (
            "# Guard: skip if data not yet loaded (will train properly in cells 39-40)\n"
            "if 'X_train_large' not in dir():\n"
            "    print('Skipping early training cell — X_train_large not loaded yet.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )
        break

# ── Cell 15: keep function def, comment out the call ──────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'launch_batch_labeler(dataset)' in c['source'] and 'def launch_batch_labeler' in c['source']:
        lines = c['source'].splitlines()
        fixed = []
        for line in lines:
            if line.strip() == 'launch_batch_labeler(dataset)':
                fixed.append('# launch_batch_labeler(dataset)  # interactive — run manually in Jupyter')
            else:
                fixed.append(line)
        c['source'] = '\n'.join(fixed)

# ── Cell 28: guard augmentation on empty dataset ──────────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'arabic_augmented_dataset.npz' in c['source'] and 'ImageDataGenerator' in c['source']:
        c['source'] = (
            "import os\n"
            "if not os.path.exists('arabic_handwritten_dataset.npz'):\n"
            "    print('No labeled dataset found — skipping augmentation.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )

# ── Cell 30: guard transfer learning on empty augmented dataset ────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'arabic_augmented_dataset.npz' in c['source'] and 'MobileNetV2' in c['source']:
        c['source'] = (
            "import os\n"
            "if not os.path.exists('arabic_augmented_dataset.npz'):\n"
            "    print('No augmented dataset found — skipping transfer learning.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )

# ── Cell 32: guard fine-tuning ─────────────────────────────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'Fine-tune' in c['source'] and 'base_model.trainable' in c['source']:
        c['source'] = (
            "if 'model_tl' not in dir():\n"
            "    print('No transfer learning model — skipping fine-tuning.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )

# ── Cell 35: guard small-dataset CNN on empty data ─────────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'arabic_handwritten_dataset.npz' in c['source'] and 'data_augmentation' in c['source']:
        c['source'] = (
            "import os\n"
            "if not os.path.exists('arabic_handwritten_dataset.npz'):\n"
            "    print('No labeled dataset — skipping small-dataset CNN training.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )

# ── Cell 37: guard evaluation on missing history/model ─────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'history_tl' in c['source'] and 'classification_report' in c['source']:
        c['source'] = (
            "if 'history_tl' not in dir() or 'model' not in dir():\n"
            "    print('Models not trained yet — skipping evaluation.')\n"
            "else:\n"
            + '\n'.join('    ' + line for line in c['source'].splitlines())
        )

# ── Cell 46: fix duplicate 'ر' at index 18, should be 'ع' ─────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'batch_auto_label_recursive' in c['source'] and "'ر', 'ف'" in c['source']:
        c['source'] = c['source'].replace("'ر', 'ف'", "'ع', 'ف'")
        print("Fixed arabic_alphabet bug: index 18 ra -> ain")

# ── Cells 47-50: replace google.colab drive with local path ───────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'google.colab' in c['source']:
        c['source'] = (
            "# Google Colab drive mount — not needed locally\n"
            "# Set your local dataset path below:\n"
            "# unlabeled_dir = 'path/to/your/200k/word/images'\n"
            "print('Running locally — set unlabeled_dir manually if you want to run auto-labeling.')"
        )

# ── Cell 48: auto-labeling call on drive path ──────────────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and "unlabeled_dir = '/content/drive" in c['source']:
        c['source'] = (
            "# Uncomment and set your local path to run auto-labeling:\n"
            "# unlabeled_dir = r'C:/path/to/your/word/images'\n"
            "# labels_df = batch_auto_label_recursive(unlabeled_dir)\n"
            "# display(labels_df.head())\n"
            "print('Auto-labeling ready — set unlabeled_dir above to run.')"
        )

# ── Cell 53: install ultralytics inline ───────────────────────────────────────
for c in cells:
    if c['cell_type'] == 'code' and 'pip install ultralytics' in c['source']:
        c['source'] = (
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', '-q'])\n"
            "from ultralytics import RTDETR\n"
            "import cv2, matplotlib.pyplot as plt\n"
            "print('Ultralytics ready.')"
        )

# Clear all outputs so notebook reruns cleanly
for c in cells:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None

nb['cells'] = cells

with open('AHR_DHAMER_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Fixed notebook written to AHR_DHAMER_fixed.ipynb")
print(f"Total cells: {len(cells)}")
