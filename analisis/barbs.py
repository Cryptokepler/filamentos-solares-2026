"""¿Nos comemos los barbs? (31 oct 2026, sin GPU)

Uso:
    python3 analisis/barbs.py <carpeta MAGFiLO_1.0_Kaggle_2026> <candidatos_examen.json> [umbral]

Los barbs son las ramitas finas que salen del cuerpo del filamento. Para cada filamento de un experto se separa su
"parte fina": lo que desaparece al hacer una apertura morfológica con un disco de radio r (todo lo que mide menos de
2r+1 px de ancho). Después se mira, en las parejas que casan (IoU > 0,5), qué fracción de esa parte fina cubre:
  - nuestra silueta (YOLO + refinador, umbral de la ronda 5),
  - la silueta de OTRO experto en la misma imagen (la referencia humana).
También se compara cuánta parte fina tiene cada silueta respecto a su área (las nuestras, ¿son más "redondas"?).
"""
import json
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import limpiar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

RADIOS = (2, 4, 6)


def nuestras(cands, t):
    confs, rles = limpiar(cands)
    sel = [r for c, r in zip(confs, rles) if c >= t]
    if len(sel) > 1:
        _, et = connected_components(iou(sel, sel) > 0, directed=False)
        sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
    return sel


def recorte(a, b, margen=20):
    """Máscaras de a y b en la caja que las contiene a las dos (más rápido que a 2048)."""
    ma, mb = mask_utils.decode(a), mask_utils.decode(b)
    ys, xs = np.nonzero(ma | mb)
    y0, y1 = max(ys.min() - margen, 0), ys.max() + margen + 1
    x0, x1 = max(xs.min() - margen, 0), xs.max() + margen + 1
    return ma[y0:y1, x0:x1], mb[y0:y1, x0:x1]


def fina(m, r):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return m & ~cv2.morphologyEx(m, cv2.MORPH_OPEN, k)


def parejas(pred, real):
    if not pred or not real:
        return []
    m = iou(pred, real)
    return [(pred[i], real[j]) for i, j in zip(*np.nonzero(m > 0.5))]


def medir(pares):
    """Para cada radio: (fracción de la parte fina del experto cubierta, parte fina propia / área propia, n)."""
    cub = {r: [0, 0] for r in RADIOS}
    prop = {r: [0, 0] for r in RADIOS}
    ref = {r: [0, 0] for r in RADIOS}
    for p, e in pares:
        mp, me = recorte(p, e)
        for r in RADIOS:
            fe = fina(me, r)
            cub[r][0] += int((fe & mp).sum())
            cub[r][1] += int(fe.sum())
            prop[r][0] += int(fina(mp, r).sum())
            prop[r][1] += int(mp.sum())
            ref[r][0] += int(fe.sum())
            ref[r][1] += int(me.sum())
    return {r: (cub[r][0] / max(cub[r][1], 1), prop[r][0] / max(prop[r][1], 1), ref[r][0] / max(ref[r][1], 1))
            for r in RADIOS}, len(pares)


if __name__ == "__main__":
    raiz, ruta = sys.argv[1], sys.argv[2]
    t = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
    cand = json.load(open(ruta))
    anot = cargar_anotaciones(raiz)
    nos, hum = [], []
    for n in sorted(cand):
        expertos = reales_rle(anot[n])
        pred = nuestras(cand[n], t)
        varios = len(expertos) > 1
        for a, ea in enumerate(expertos):
            if not varios:
                continue  # solo imágenes con 2-3 expertos, para comparar contra la referencia humana
            nos += parejas(pred, ea)
            for b, eb in enumerate(expertos):
                if b != a:
                    hum += parejas(eb, ea)
    for nombre, pares in (("Nuestra silueta", nos), ("Otro experto", hum)):
        res, k = medir(pares)
        print(f"\n=== {nombre} frente a un experto ({k} parejas que casan, imágenes con 2-3 expertos) ===")
        for r in RADIOS:
            c, p, e = res[r]
            print(f"  partes de menos de {2 * r + 1:2d} px de ancho: cubre {c:5.1%} de las del experto | "
                  f"parte fina propia {p:5.1%} del área (la del experto: {e:5.1%})")
