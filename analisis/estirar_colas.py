"""Estirar las colas siguiendo la vena oscura (idea del 31 oct 2026, sin GPU).

Uso (FUERZAS=0.5,1,2 por defecto: valores de a a probar):
    python3 analisis/estirar_colas.py <carpeta MAGFiLO> <candidatos_examen.json> <cache_venas_dir> [umbral_conf]

El refinador no ve las colas finas de los extremos. Aquí, cada silueta elegida crece paso a paso (1 px por paso, hasta
L pasos) solo por píxeles que el filtro de Frangi marca como "vena oscura" (fuerza >= a × la mediana de la vena
dentro de la propia silueta) y que son más oscuros que el fondo cercano. No puede entrar en otra silueta. Se prueban
varios a y L y se mide de forma honesta (ajuste elegido en una mitad del examen, medido en la otra), además de cuánto
recupera de las partes finas del experto.
"""
import itertools
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.filters import frangi

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.barbs import fina, medir, nuestras, parejas  # noqa: E402
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from filamentos import clasico  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

TAM = 2048
MARGEN = 80
CRUZ = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)


def mapa_venas(img):
    """Fuerza de 'vena oscura' a 2048 (Frangi, rayas oscuras de 2-8 px de ancho) y oscuridad respecto al fondo."""
    x = img.astype(np.float32) / 255
    v = frangi(x, sigmas=(1, 2, 3, 4), black_ridges=True).astype(np.float32)
    fondo = cv2.medianBlur(img, 51).astype(np.float32)
    oscuro = (fondo - img.astype(np.float32)) > 3
    cx, cy, r = clasico.disco_solar(img)
    yy, xx = np.mgrid[:TAM, :TAM]
    disco = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.97 * r) ** 2
    return v, oscuro & disco


def estirar(m, otras, v, oscuro, a, pasos):
    ys, xs = np.nonzero(m)
    y0, y1 = max(ys.min() - MARGEN, 0), min(ys.max() + MARGEN + 1, TAM)
    x0, x1 = max(xs.min() - MARGEN, 0), min(xs.max() + MARGEN + 1, TAM)
    sm = m[y0:y1, x0:x1].astype(np.uint8)
    sv = v[y0:y1, x0:x1]
    ref = np.median(sv[sm > 0])
    if ref <= 0:
        return m
    via = ((sv >= a * ref) & oscuro[y0:y1, x0:x1] & ~otras[y0:y1, x0:x1]).astype(np.uint8)
    actual = sm.copy()
    for _ in range(pasos):
        nuevo = cv2.dilate(actual, CRUZ) & (via | sm)
        if (nuevo == actual).all():
            break
        actual = nuevo
    out = m.copy()
    out[y0:y1, x0:x1] |= actual.astype(bool)
    return out


def nota(sel_por_img, reales, imgs):
    tp = fp = fn = 0
    suma = 0.0
    for n in imgs:
        sel = sel_por_img[n]
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
    den = tp + 0.5 * fp + 0.5 * fn
    return (suma / den if den else 0.0), tp, fp, fn


if __name__ == "__main__":
    raiz, ruta, cache = sys.argv[1:4]
    t = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    os.makedirs(cache, exist_ok=True)
    cand = json.load(open(ruta))
    anot = cargar_anotaciones(raiz)
    imgs = sorted(cand)
    reales = {n: reales_rle(anot[n]) for n in imgs}
    base = {n: nuestras(cand[n], t) for n in imgs}
    fuerzas = tuple(float(x) for x in os.environ.get("FUERZAS", "0.5,1,2").split(","))
    recetas = [(None, 0)] + list(itertools.product(fuerzas, (10, 25, 50)))
    salida = {r: {} for r in recetas}
    for k, n in enumerate(imgs):
        f = f"{cache}/{os.path.splitext(n)[0]}.npz"
        if os.path.exists(f):
            z = np.load(f)
            v, oscuro = z["v"].astype(np.float32), z["oscuro"]
        else:
            img = np.asarray(Image.open(f"{raiz}/train/train_images/{n}").convert("L"), np.uint8)
            v, oscuro = mapa_venas(img)
            np.savez_compressed(f, v=v.astype(np.float16), oscuro=oscuro)
        ms = [mask_utils.decode(r).astype(bool) for r in base[n]]
        todas = np.logical_or.reduce(ms) if ms else np.zeros((TAM, TAM), bool)
        for r in recetas:
            if r[0] is None:
                salida[r][n] = base[n]
                continue
            nuevas = []
            for m in ms:
                otras = cv2.dilate((todas & ~m).astype(np.uint8), CRUZ, iterations=2).astype(bool)
                nuevas.append(estirar(m, otras, v, oscuro, *r))
            salida[r][n] = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in nuevas]
        if k % 20 == 0:
            print(f"{k + 1}/{len(imgs)} imágenes", flush=True)

    rng = np.random.default_rng(0)
    orden = list(imgs)
    rng.shuffle(orden)
    mA, mB = orden[:60], orden[60:]
    filas = []
    for r in recetas:
        filas.append((r, nota(salida[r], reales, imgs), nota(salida[r], reales, mA)[0], nota(salida[r], reales, mB)[0]))
    ra = max(filas, key=lambda x: x[2])
    rb = max(filas, key=lambda x: x[3])
    print(f"\nHonesto (receta elegida en una mitad, medida en la otra): {(ra[3] + rb[2]) / 2:.4f} "
          f"(recetas {ra[0]} y {rb[0]})")
    print("\nreceta (a, pasos) | PQ | aciertos inventados escapados | cubre <9 px del experto | parte fina propia")
    for r, (pq, tp, fp, fn), _, _ in filas:
        pares = []
        for n in imgs:
            if len(reales[n]) > 1:
                for e in reales[n]:
                    pares += parejas(salida[r][n], e)
        res, _ = medir(pares)
        c, p, _ = res[4]
        print(f"  {str(r):12s} | {pq:.4f} | {tp} {fp} {fn} | {c:5.1%} | {p:5.1%}")
