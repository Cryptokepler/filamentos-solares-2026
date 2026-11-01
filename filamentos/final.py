"""Final model: two YOLO11-seg detectors + a U-Net refiner + a 2-of-2 jury.

This is the exact pipeline behind our best Kaggle submission (public PQ 0.39, local exam PQ 0.4607).
Everything else in the repository is the experiment history that led here (see docs/).

Per image (2048 x 2048 GONG H-alpha, grayscale):

1. **Detect.** Two YOLO11m-seg models look at the image at 1024 px and propose filaments generously
   (confidence >= 0.1):
   - ``yolo_r5``: trained on the plain image (round 5);
   - ``yolo_r7``: trained on three "views" of the image, one per channel: original, local contrast (CLAHE)
     and a black-hat "dark and thin" map (round 7, see ``yolo.tres_vistas``).
2. **Refine.** Each proposal is upsampled to 2048 and redrawn by a U-Net (ResNet-18 encoder) on a square
   crop around it (1.8x its size, resized to 256 px, 4 flips averaged, threshold 0.5). We keep the connected
   part that touches the detector's mask.
3. **Clean.** Per model: drop proposals whose mean distance to the disk centre is above 0.95 solar radii
   (limb artefacts) and near-duplicates (IoU > 0.5 with a more confident one).
4. **Jury.** Proposals of the two models that overlap with IoU > 0.5 form one *case*. A case is kept when
   both models see it with confidence >= 0.2; its mask is the most confident proposal's refined mask.
5. **No overlaps.** Kept masks that touch are merged, so every pixel belongs to at most one filament
   (the scorer requires it).
"""
import csv
import os

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

from . import candidatos as C
from . import detector as D
from . import yolo as Y

RES = 1024            # detector input size
CONF_MIN = 0.1        # detector proposals kept
REFINER_THRESHOLD = 0.5
MAX_RADIUS = 0.95     # in solar radii: beyond this the limb causes false filaments
DUPLICATE_IOU = 0.5
JURY_IOU = 0.5
JURY_CONF = 0.2       # t: minimum confidence for a model's vote
JURY_VOTES = 2        # k: votes needed (2 of 2)

WEIGHTS = {"yolo_r5": "yolo_r5.pt", "yolo_r7": "yolo_r7_three_views.pt", "refiner": "refiner.pt"}


def _iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mask_utils.iou(a, b, [0] * len(b)))


def load_models(weights_dir, device="cpu"):
    """Load the two detectors and the refiner from ``weights_dir`` (the repository's ``weights/`` folder)."""
    import segmentation_models_pytorch as smp
    import torch
    from ultralytics import YOLO

    refiner = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1).to(device)
    refiner.load_state_dict(torch.load(os.path.join(weights_dir, WEIGHTS["refiner"]), map_location=device))
    refiner.eval()
    detectors = {
        "yolo_r5": Y.detector(YOLO(os.path.join(weights_dir, WEIGHTS["yolo_r5"])), RES, CONF_MIN, vistas=False),
        "yolo_r7": Y.detector(YOLO(os.path.join(weights_dir, WEIGHTS["yolo_r7"])), RES, CONF_MIN, vistas=True),
    }
    return detectors, refiner


def proposals(detect, refiner, image, device="cpu"):
    """Steps 1-3 for one detector. image: uint8 [2048, 2048]. Returns (confidences, refined RLE masks)."""
    import torch

    small = np.asarray(Image.fromarray(image).resize((RES, RES), Image.BILINEAR), np.uint8)
    radius = np.asarray(Image.fromarray(C.radio(small)).resize((D.TAM, D.TAM), Image.BILINEAR))
    with torch.autocast(device, enabled=device == "cuda"):
        masks, conf = detect(small)
    big = [D.a_2048(m) for m in masks]
    keep = [i for i, m in enumerate(big) if m.any() and radius[m].mean() <= MAX_RADIUS]
    big, conf = [big[i] for i in keep], conf[keep]
    with torch.autocast(device, enabled=device == "cuda"):
        refined = D.refinar(refiner, image, big, device, umbral=REFINER_THRESHOLD, unir=False)
    order = np.argsort(-conf, kind="stable")
    conf = conf[order]
    rles = [D.a_rle(refined[i]) for i in order]
    if not rles:
        return np.zeros(0), []
    dup = np.triu(_iou(rles, rles) > DUPLICATE_IOU, 1).any(0)
    return conf[~dup], [r for r, d in zip(rles, dup) if not d]


def jury(per_model, t=JURY_CONF, k=JURY_VOTES):
    """Steps 4-5. per_model: list of (confidences, RLEs), one per detector. Returns the final RLE masks."""
    pool = [(m, c, r) for m, (cs, rs) in enumerate(per_model) for c, r in zip(cs, rs)]
    if not pool:
        return []
    rles = [r for _, _, r in pool]
    model = np.array([m for m, _, _ in pool])
    link = (_iou(rles, rles) > JURY_IOU) & (model[:, None] != model[None, :])
    _, case = connected_components(link, directed=False)
    kept = []
    for g in range(case.max() + 1):
        idx = np.flatnonzero(case == g)
        votes = np.zeros(len(per_model))
        for i in idx:
            votes[model[i]] = max(votes[model[i]], pool[i][1])
        n = int((votes >= t).sum())
        if n >= k:
            best = max(idx, key=lambda i: pool[i][1])
            kept.append((n + votes.max(), pool[best][2]))
    kept = [r for _, r in sorted(kept, key=lambda x: -x[0])]
    if len(kept) > 1:  # merge masks that touch: no pixel may belong to two filaments
        _, group = connected_components(_iou(kept, kept) > 0, directed=False)
        kept = [mask_utils.merge([kept[j] for j in np.flatnonzero(group == g)]) for g in range(group.max() + 1)]
    return kept


def predict(image, detectors, refiner, device="cpu"):
    """Whole pipeline for one image (uint8 [2048, 2048]). Returns a list of RLE masks (pycocotools format)."""
    return jury([proposals(detectors[name], refiner, image, device) for name in ("yolo_r5", "yolo_r7")])


def load_image(path):
    return np.asarray(Image.open(path).convert("L"), np.uint8)


def write_submission(predictions, path):
    """predictions: {image file name: [RLE]} -> Kaggle CSV (filament_id, segmentation_rle)."""
    rows = []
    for name in sorted(predictions):
        image_id = os.path.splitext(name)[0]
        for n, r in enumerate(predictions[name], start=1):
            c = r["counts"]
            rows.append((f"{image_id}_{n}", c.decode("ascii") if isinstance(c, bytes) else c))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    return len(rows)
