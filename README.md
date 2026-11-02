# Solar filament segmentation 2026

Our solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData Cup 2026, MAGFiLO dataset, GONG H-alpha images at 2048 x 2048).

**Author:** Jhonatan Venegas. Code written with the help of an AI assistant (Claude); every experiment was
discussed, chosen and checked by the author (see the experiment diary in `docs/`).

## Results

| | Local exam (121 train images, never used for training) | Kaggle public |
|---|---|---|
| Expert vs. expert (296 images with 2-3 annotators) | 0.340 | — |
| Classical baseline (dark regions + rules) | 0.098 | — |
| U-Net segmentation (round 1) | 0.384 | 0.34 |
| Mask R-CNN detector + U-Net refiner (round 3) | 0.4405 | 0.38 |
| **Final model: 2 YOLO11-seg + refiner + 2-of-2 jury** | **0.4607** | **0.39** |

On the local exam the final model finds 892 filaments, invents 382 and misses 475 (pooled over every expert).
Its matched masks have a mean IoU (SQ) of 0.68, above the 0.64 two experts reach with each other.

## Final model

Code: [`filamentos/final.py`](filamentos/final.py). Demo: [`notebooks/demo_final_model.ipynb`](notebooks/demo_final_model.ipynb).
Weights: [`weights/`](weights/).

1. **Detect.** Two YOLO11m-seg models look at the image at 1024 px and propose filaments (confidence >= 0.1).
   One sees the plain image; the other sees three views of it, one per channel: the original, local
   contrast (CLAHE) and a black-hat "dark and thin" map (`filamentos/yolo.py`, `tres_vistas`).
2. **Refine.** Each proposal is redrawn at full resolution by a U-Net (ResNet-18 encoder) on a square crop
   around it (1.8x its size, 256 px, 4 flips averaged). Only the connected part touching the detector's
   mask is kept.
3. **Clean.** Proposals near the limb (mean radius > 0.95 R) and near-duplicates (IoU > 0.5) are dropped.
4. **Jury.** Proposals from both detectors that overlap (IoU > 0.5) are one case. A case is kept only when
   both detectors see it with confidence >= 0.2; its mask is the most confident one.
5. **No overlaps.** Kept masks that touch are merged, so no pixel belongs to two filaments.

Only the test image is used at inference: no metadata and no neighbouring frames. No external data or
labels were used; the detectors start from Ultralytics' public COCO weights and the refiner's encoder from
ImageNet weights.

| File | What | Size |
|---|---|---|
| `weights/yolo_r5.pt` | YOLO11m-seg, plain image (round 5) | 45 MB |
| `weights/yolo_r7_three_views.pt` | YOLO11m-seg, three views (round 7) | 45 MB |
| `weights/refiner.pt` | U-Net ResNet-18 refiner (round 3) | 57 MB |

## Run it

```
pip install -r requirements.txt
kaggle competitions download -c filament-segmentation-2026 && unzip filament-segmentation-2026.zip -d data
MAGFILO=data/MAGFiLO_1.0_Kaggle_2026 jupyter nbconvert --to notebook --execute notebooks/demo_final_model.ipynb
```

The notebook writes `submission.csv` for the 180 test images and, if the training data is present, scores
the local exam (PQ, matched / invented / missed, IoU and Dice distributions, one-to-many and many-to-one).
A CPU is enough: the 180 test images took 43 min on 4 cores (about 14 s per image). The competition data is not redistributed in this repository.

## Training

All models were trained on Kaggle (one T4 GPU) with 586 of the 707 training images; the other 121 are the
fixed local exam (`md5(file name) % 100 < 15`). Each expert's annotation is a separate training example.

| Model | Notebook (generator script) | Time |
|---|---|---|
| Refiner | `notebooks/entrenar_detector.ipynb` (`generar_notebook_detector.py`) | 5 h with the Mask R-CNN of round 3 |
| YOLO, plain image | `notebooks/entrenar_yolo.ipynb` (`generar_notebook_yolo.py`) | 2.7 h |
| YOLO, three views | `notebooks/ronda7_tres_vistas.ipynb` (`generar_notebook_ronda7.py`) | 2.9 h |

The notebooks embed the code of `filamentos/` and are rebuilt by their generator scripts.

## What is where

- `filamentos/`: the library. `final.py` (final model), `yolo.py` (YOLO detector and image views),
  `detector.py` (Mask R-CNN and the refiner), `candidatos.py` (proposals and their features),
  `metrica.py` (Panoptic Quality as the competition defines it, plus fragmentation / merge counts),
  `datos.py` (MAGFiLO reading, polygons / masks / RLE), `clasico.py` (classical baseline, solar disk).
- `analisis/`: every measurement behind a decision, including the negative results.
- `docs/`: the experiment diary (`experimentos.md`, written before and after each run), a round-by-round
  summary (`resumen-rondas.md`), annotation rules observed in the data and a knowledge base with sources.
- `visor/`: a small web viewer of images, expert annotations and our predictions.

The diary and most code comments are in Spanish, the author's language; this README, the final model and
the demo are in English.

## What did not help (measured)

Test-time flips (−0.008), a small-filament specialist at 2048 px (0.439, finds 88 more but invents ~170),
more images of the same kind (learning curve flat after ~300 images), hand-written and learned judges,
juries of identical YOLOs, a Frangi "vessel" view as third judge, higher refiner thresholds, and growing
mask tails along dark ridges. Details and numbers are in `docs/experimentos.md`.

## Reproducibility check

Running `filamentos/final.py` with the weights in `weights/` on the 180 test images (CPU) gives 1,210 filaments
in 177 images. Our submitted file (trained and predicted on a Kaggle GPU) has 1,212. Every filament from the
CPU run has a match in the submitted file: 1,161 with IoU > 0.95 and 49 with IoU between 0.5 and 0.95. The small
differences come from GPU half-precision versus CPU full precision near the 0.2 confidence threshold.
