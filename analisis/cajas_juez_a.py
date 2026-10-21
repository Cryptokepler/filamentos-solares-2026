"""Cajas y siluetas recortadas de los candidatos (para el juez A), en el mismo orden que tribunal_sim.

Uso: python3 analisis/cajas_juez_a.py <carpeta con candidatos_*.json> <salida>
Guarda <salida>/cajas.json ({imagen: [[x0,y0,x1,y1], ...]}) y <salida>/siluetas_<grupo>.npy.
"""
import json
import sys

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import iou  # noqa: E402
from filamentos.candidatos import de_texto  # noqa: E402

TAM, LADO, CONF_MIN = 2048, 128, 0.3


def caja(rle, margen=1.8, minimo=96):
    """Igual que detector.caja_cuadrada pero desde la caja del RLE (sin decodificar)."""
    x, y, w, h = mask_utils.toBbox(rle)
    cx, cy = x + w / 2, y + h / 2
    lado = int(np.ceil(max(w, h) * margen))
    lado = min(max(lado, minimo), TAM)
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x0, y0 = min(max(x0, 0), TAM - lado), min(max(y0, 0), TAM - lado)
    return [x0, y0, x0 + lado, y0 + lado]


def orden_sin_repetidos(cands):
    """Mismo orden y mismos descartes que tribunal_sim.preparar_imagen."""
    orden = sorted(range(len(cands)), key=lambda i: -cands[i]["conf"])
    cands = [cands[i] for i in orden]
    rles = [de_texto(c["rle_ref"]) for c in cands]
    dup = np.triu(iou(rles, rles) > 0.5, 1).any(0) if cands else []
    return [c for c, d in zip(cands, dup) if not d]


if __name__ == "__main__":
    carpeta, salida = sys.argv[1], sys.argv[2]
    cajas = {}
    for grupo in ("examen", "estudio"):
        with open(f"{carpeta}/candidatos_{grupo}.json") as fh:
            todos = json.load(fh)
        sil = []
        for img in sorted(todos):
            cs = orden_sin_repetidos(todos[img])
            cajas[img] = []
            for c in cs:
                if c["conf"] < CONF_MIN:
                    cajas[img].append(None)
                    continue
                r = de_texto(c["rle"])
                b = caja(r)
                cajas[img].append(b)
                m = mask_utils.decode(r)[b[1]:b[3], b[0]:b[2]] * 255
                sil.append(np.asarray(Image.fromarray(m).resize((LADO, LADO), Image.NEAREST)))
        np.save(f"{salida}/siluetas_{grupo}.npy", np.stack(sil).astype(np.uint8))
        print(grupo, len(sil), "recortes")
    with open(f"{salida}/cajas.json", "w") as fh:
        json.dump(cajas, fh)
