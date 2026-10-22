"""¿Cuánto se puede ganar dibujando mejor el borde? (22 oct 2026, sin GPU)

Uso:
    python3 analisis/techo_borde.py <carpeta MAGFiLO_1.0_Kaggle_2026> <candidatos_yolo_examen.json>

A) Nuestro dibujo en el examen (YOLO ronda 5 + refinador, umbral 0,25): ¿es sistemáticamente más gordo o más
   flaco que el de los expertos? Se prueba engordarlo o adelgazarlo k píxeles y se mide la nota.
B) Borde "promedio" con los dibujos de los expertos (imágenes con 3 expertos, todo train): para cada filamento de
   un experto, ¿se le parece más el dibujo de otro experto o una combinación de los otros dos (intersección,
   unión)? Es el techo de la idea "aprender el borde donde coinciden".
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

TAM = 2048


def morfo(rle, k):
    """Engorda (k > 0) o adelgaza (k < 0) una máscara k píxeles, trabajando en su recuadro."""
    if k == 0:
        return rle
    x, y, w, h = (int(v) for v in mask_utils.toBbox(rle))
    p = abs(k) + 2
    x0, y0, x1, y1 = max(x - p, 0), max(y - p, 0), min(x + w + p, TAM), min(y + h + p, TAM)
    m = mask_utils.decode(rle)[y0:y1, x0:x1]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1, 2 * abs(k) + 1))
    m = cv2.dilate(m, ker) if k > 0 else cv2.erode(m, ker)
    full = np.zeros((TAM, TAM), np.uint8)
    full[y0:y1, x0:x1] = m
    return mask_utils.encode(np.asfortranarray(full))


def seleccion(cands, umbral):
    conf, rles = limpiar(cands)
    sel = [r for c, r in zip(conf, rles) if c >= umbral]
    if len(sel) > 1:
        _, et = connected_components(iou(sel, sel) > 0, directed=False)
        sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
    return sel


def pq(sel_por_img, reales):
    tp = fp = fn = 0
    suma = 0.0
    ratios = []
    for n, sel in sel_por_img.items():
        for rs in reales[n]:
            if not sel or not rs:
                fp += len(sel)
                fn += len(rs)
                continue
            m = iou(sel, rs)
            hit = m > 0.5
            tp += int(hit.any(0).sum())
            suma += float(m[hit].sum())
            fn += int((~hit.any(0)).sum())
            fp += len(sel) - int(hit.any(1).sum())
            a_sel = mask_utils.area(sel)
            a_rs = mask_utils.area(rs)
            for i, j in zip(*np.nonzero(hit)):
                ratios.append(a_sel[i] / a_rs[j])
    den = tp + 0.5 * fp + 0.5 * fn
    return suma / den, suma / max(tp, 1), tp, fp, fn, np.array(ratios)


def parte_a(raiz, fcand, anot):
    cand = json.load(open(fcand))
    imgs = sorted(cand)
    reales = {n: reales_rle(anot[n]) for n in imgs}
    base = {n: seleccion(cand[n], 0.25) for n in imgs}
    print("A) Nuestro dibujo en el examen (YOLO + refinador, umbral 0,25)")
    for k in (-3, -2, -1, 0, 1, 2, 3):
        sel = {n: [morfo(r, k) for r in s] for n, s in base.items()}
        p, sq, tp, fp, fn, ratios = pq(sel, reales)
        print(f"  {'+' if k > 0 else ''}{k} px: PQ {p:.4f} SQ {sq:.3f} | aciertos {tp} inventados {fp} escapados {fn}"
              f" | área nuestra / experto: mediana {np.median(ratios):.2f}")


def parte_b(anot):
    print("\nB) Borde promedio entre expertos (imágenes con 3 expertos)")
    uno, inter, union, n = [], [], [], 0
    for nombre, por_exp in anot.items():
        if len(por_exp) != 3:
            continue
        n += 1
        exps = reales_rle({k: v for k, v in por_exp.items()})
        for e in range(3):
            o1, o2 = [exps[i] for i in range(3) if i != e]
            if not exps[e] or not o1 or not o2:
                continue
            m1, m2 = iou(exps[e], o1), iou(exps[e], o2)
            for f in range(len(exps[e])):
                j1, j2 = m1[f].argmax(), m2[f].argmax()
                if m1[f, j1] > 0.5 and m2[f, j2] > 0.5:  # los otros dos dibujaron este filamento
                    a, b = o1[j1], o2[j2]
                    uno.append((m1[f, j1] + m2[f, j2]) / 2)
                    inter.append(iou([exps[e][f]], [mask_utils.merge([a, b], intersect=True)])[0, 0])
                    union.append(iou([exps[e][f]], [mask_utils.merge([a, b])])[0, 0])
    print(f"  {n} imágenes, {len(uno)} filamentos dibujados por los tres")
    print(f"  parecido (IoU medio) con el dibujo de OTRO experto:        {np.mean(uno):.3f}")
    print(f"  parecido con la INTERSECCIÓN de los otros dos (lo común):  {np.mean(inter):.3f}")
    print(f"  parecido con la UNIÓN de los otros dos:                   {np.mean(union):.3f}")


if __name__ == "__main__":
    raiz, fcand = sys.argv[1:3]
    anot = cargar_anotaciones(raiz)
    parte_b(anot)
    parte_a(raiz, fcand, anot)
