"""Builds notebooks/demo_final_model.ipynb, the end-to-end demo of the final model (English, for the jury).

Run from the repository root:
    python3 notebooks/generar_notebook_demo.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Solar filament segmentation: final model demo

IEEE BigData Cup 2026, *Filament Segmentation* (MAGFiLO / GONG H-alpha, 2048 x 2048).

This notebook loads the weights stored in this repository (`weights/`) and runs the **final model** on the
test images, writing `submission.csv`. It is the same model as our best Kaggle submission
(public PQ 0.39; local exam PQ 0.4607).

**Pipeline** (code in `filamentos/final.py`):

1. **Detect**: two YOLO11m-seg models at 1024 px propose filaments (confidence >= 0.1). One sees the plain
   image (round 5); the other sees three views of it, one per channel: original, local contrast (CLAHE) and
   a black-hat "dark and thin" map (round 7).
2. **Refine**: every proposal is redrawn at full 2048 px resolution by a U-Net (ResNet-18 encoder) on a
   square crop around it (4 flips averaged).
3. **Clean**: drop proposals near the limb (mean radius > 0.95 R) and near-duplicates (IoU > 0.5).
4. **Jury (2 of 2)**: proposals of both detectors that overlap (IoU > 0.5) form a case; a case is kept only if
   both detectors see it with confidence >= 0.2.
5. **No overlaps**: kept masks that touch are merged, so every pixel belongs to at most one filament.

Only the test image itself is used at inference (no metadata, no neighbouring frames).

**How to run**

- *Locally*: `pip install -r requirements.txt`, then set `MAGFILO` to the folder that contains `train/` and
  `test/` from the competition data (or `TEST_DIR` to a folder of test images) and run all cells. A CPU is
  enough: on 4 cores the 180 test images take about 43 min (two detectors + refiner).
- *Kaggle*: add the competition as Input, turn Internet on, and run the first cell, which clones this
  repository and installs the requirements.
""")

code("""
# Where things are. Edit these if your folders differ.
import glob, os, subprocess, sys

REPO_URL = 'https://github.com/Cryptokepler/filamentos-solares-2026'

def find_repo():
    for d in (os.getcwd(), os.path.dirname(os.getcwd()), '/kaggle/working/filamentos-solares-2026'):
        if os.path.isfile(os.path.join(d, 'filamentos', 'final.py')):
            return d
    subprocess.run(['git', 'clone', '--depth', '1', REPO_URL], check=True)          # Kaggle / fresh machine
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                    'filamentos-solares-2026/requirements.txt'], check=True)
    return os.path.abspath('filamentos-solares-2026')

REPO = find_repo()
sys.path.insert(0, REPO)
WEIGHTS = os.path.join(REPO, 'weights')

def find_data():
    if os.environ.get('MAGFILO'):
        return os.environ['MAGFILO']
    hits = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)
    return os.path.dirname(os.path.dirname(hits[0])) if hits else os.path.join(REPO, 'data')

DATA = find_data()
TEST_DIR = os.environ.get('TEST_DIR', os.path.join(DATA, 'test', 'test_images'))
QUICK = False   # True: only 3 test images, to check that everything runs
print('repo:', REPO, '| data:', DATA, '| test images:', TEST_DIR)
""")

code("""
import time
import numpy as np, torch
from PIL import Image
from pycocotools import mask as mask_utils
from filamentos import final as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
for f in F.WEIGHTS.values():
    p = os.path.join(WEIGHTS, f)
    print(f'{f:28s} {os.path.getsize(p) / 1e6:6.1f} MB')
detectors, refiner = F.load_models(WEIGHTS, device)
print('models loaded on', device)
""")

md("""
## 1. One image, step by step

Left: the image. Right: the final masks outlined in red. Saved as `example.png`.
""")
code("""
test = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png')))
if QUICK:
    test = test[:3]
name = test[0]
image = F.load_image(os.path.join(TEST_DIR, name))
per_model = {k: F.proposals(d, refiner, image, device) for k, d in detectors.items()}
final = F.jury(list(per_model.values()))
for k, (conf, _) in per_model.items():
    print(f'{k}: {len(conf)} proposals, {(conf >= F.JURY_CONF).sum()} with confidence >= {F.JURY_CONF}')
print(f'jury (2 of 2): {len(final)} filaments')

import cv2
rgb = np.stack([image] * 3, -1).copy()
for r in final:
    m = mask_utils.decode(r)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (255, 40, 40), 2)
Image.fromarray(np.concatenate([np.stack([image] * 3, -1), rgb], 1)).resize((2048, 1024)).save('example.png')
from IPython.display import Image as Show, display
display(Show('example.png', width=700))
""")

md("## 2. All test images and `submission.csv`")
code("""
t0 = time.time()
predictions = {}
for i, name in enumerate(test):
    predictions[name] = F.predict(F.load_image(os.path.join(TEST_DIR, name)), detectors, refiner, device)
    if i % 20 == 0:
        print(f'{i + 1}/{len(test)} images, {time.time() - t0:.0f} s', flush=True)
n = F.write_submission(predictions, 'submission.csv')
print(f'{n} filaments in {sum(map(bool, predictions.values()))}/{len(test)} images -> submission.csv '
      f'({(time.time() - t0) / 60:.1f} min)')

# Checks the scorer cares about: unique ids, 2048 x 2048 masks, no pixel in two filaments
import pandas as pd
sub = pd.read_csv('submission.csv')
assert sub.filament_id.is_unique
for name, rles in predictions.items():
    if len(rles) > 1:
        assert (np.asarray(mask_utils.iou(rles, rles, [0] * len(rles))) > 0).sum() == len(rles), name
print('ok: unique ids, no overlapping masks')
""")

md("""
## 3. Local exam (optional, needs the training data)

The 121-image exam we used throughout the project (fixed split: `md5(file name) % 100 < 15`; none of these
images were used to train the detectors or the refiner). Each expert's annotation is scored separately and
pooled, like the competition metric. Besides PQ we report what the rubric looks at: matched / invented /
missed filaments, the IoU and Dice distributions of matches, and one-to-many / many-to-one relations.
""")
code("""
import hashlib
from filamentos.datos import cargar_anotaciones, mascaras_de
from filamentos.metrica import Resultado, evaluar_imagen, resumen

ann_file = os.path.join(DATA, 'train', 'MAGFiLO_1.0_Annotations_kaggle2026_train.json')
if not os.path.exists(ann_file):
    print('No training data found: skipping the local exam.')
else:
    ANN = cargar_anotaciones(DATA)
    exam = sorted(n for n in ANN if int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
                  and os.path.exists(os.path.join(DATA, 'train', 'train_images', n)))
    if QUICK:
        exam = exam[:3]
    total, t0 = Resultado(), time.time()
    for i, name in enumerate(exam):
        pred = [mask_utils.decode(r).astype(bool) for r in
                F.predict(F.load_image(os.path.join(DATA, 'train', 'train_images', name)), detectors, refiner, device)]
        for anns in ANN[name].values():          # one expert at a time
            total += evaluar_imagen(list(mascaras_de(anns)), pred)
        if i % 20 == 0:
            print(f'{i + 1}/{len(exam)} images, {time.time() - t0:.0f} s', flush=True)
    print(f'\\n{len(exam)} exam images:', resumen(total))
    print(f'matched {total.tp} | invented {total.fp} | missed {total.fn}')
    ious = np.array(total.ious)
    dice = 2 * ious / (1 + ious)
    for label, v in (('IoU', ious), ('Dice', dice)):
        q = np.percentile(v, [10, 25, 50, 75, 90])
        print(f'{label} of matches: mean {v.mean():.3f} | p10 {q[0]:.3f} p25 {q[1]:.3f} median {q[2]:.3f} '
              f'p75 {q[3]:.3f} p90 {q[4]:.3f}')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
salida = os.path.join(RAIZ, "notebooks", "demo_final_model.ipynb")
nbf.write(nb, salida)
print("->", salida)
