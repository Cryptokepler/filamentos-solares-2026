"""Jurado de YOLO sin GPU: varios YOLO votan cada candidato del examen (idea de jhonatan, 22 oct 2026).

Uso:
    python3 analisis/jurado_yolo.py <carpeta MAGFiLO_1.0_Kaggle_2026> <nombre=candidatos.json> [<nombre=candidatos.json> ...]

Por imagen, los candidatos de los distintos YOLO que se solapan (IoU > 0,5) forman un "caso". Cada YOLO vota a
favor si tiene en el caso un candidato con confianza >= t. El caso pasa si tiene al menos k votos; su silueta es la
del candidato más seguro. Se prueban jurados (combinaciones de YOLO), t y k, y se mide de forma honesta: el
ajuste se elige en una mitad del examen y se mide en la otra.
"""
import itertools
import json
import sys

import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import limpiar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402


def casos(por_modelo):
    """Agrupa los candidatos de varios modelos de una imagen. Devuelve lista de (confs por modelo, rle del mejor)."""
    todos = [(m, c, r) for m, (cs, rs) in enumerate(por_modelo) for c, r in zip(cs, rs)]
    if not todos:
        return []
    rles = [r for _, _, r in todos]
    mod = np.array([m for m, _, _ in todos])
    enlace = (iou(rles, rles) > 0.5) & (mod[:, None] != mod[None, :])
    _, et = connected_components(enlace, directed=False)
    out = []
    for k in range(et.max() + 1):
        idx = np.flatnonzero(et == k)
        conf = np.zeros(len(por_modelo))
        for i in idx:
            conf[mod[i]] = max(conf[mod[i]], todos[i][1])
        mejor = max(idx, key=lambda i: todos[i][1])
        out.append((conf, todos[mejor][2]))
    return out


def elegir(cs, receta):
    tipo, p = receta
    sel = []
    for conf, r in cs:
        if tipo == "votos":
            v = (conf >= p["t"]).sum()
            if v >= p["k"]:
                sel.append((v + conf.max(), r))
        else:  # media de confianzas (0 si un YOLO no lo ve)
            if conf.mean() >= p["t"]:
                sel.append((conf.mean(), r))
    sel.sort(key=lambda x: -x[0])
    return [r for _, r in sel]


def nota(datos, reales, receta, imagenes):
    """PQ del jurado: une los elegidos que se tocan (como el juez) y compara con cada experto."""
    tp = fp = fn = 0
    suma = 0.0
    for n in imagenes:
        sel = elegir(datos[n], receta)
        if len(sel) > 1:
            _, et = connected_components(iou(sel, sel) > 0, directed=False)
            sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
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
    return suma / den if den else 0.0, tp, fp, fn


if __name__ == "__main__":
    raiz = sys.argv[1]
    modelos = dict(a.split("=", 1) for a in sys.argv[2:])
    cand = {k: json.load(open(v)) for k, v in modelos.items()}
    nombres = list(cand)
    imgs = sorted(set.intersection(*(set(c) for c in cand.values())))
    anot = cargar_anotaciones(raiz)
    reales = {n: reales_rle(anot[n]) for n in imgs}
    limpio = {n: {k: limpiar(cand[k][n]) for k in nombres} for n in imgs}
    print(f"{len(imgs)} imágenes del examen; modelos: {nombres}")

    rng = np.random.default_rng(0)
    orden = list(imgs)
    rng.shuffle(orden)
    mA, mB = orden[:60], orden[60:]

    jurados = [(k,) for k in nombres] + [c for n in range(2, len(nombres) + 1) for c in itertools.combinations(nombres, n)]
    for jur in jurados:
        datos = {n: casos([limpio[n][k] for k in jur]) for n in imgs}
        filas = []
        recetas = [("votos", dict(t=t, k=k)) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4) for k in range(1, len(jur) + 1)]
        if len(jur) > 1:
            recetas += [("media", dict(t=t)) for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35)]
        for r in recetas:
            filas.append((r, nota(datos, reales, r, imgs), nota(datos, reales, r, mA)[0], nota(datos, reales, r, mB)[0]))
        filas.sort(key=lambda x: -x[1][0])
        ra = max(filas, key=lambda x: x[2])
        rb = max(filas, key=lambda x: x[3])
        (pq, tp, fp, fn), r = filas[0][1], filas[0][0]
        print(f"\n=== Jurado {'+'.join(jur)} ===")
        print(f"  mejor {pq:.4f} (aciertos {tp}, inventados {fp}, escapados {fn}) con {r}")
        print(f"  honesto (elegido en una mitad, medido en la otra): {(ra[3] + rb[2]) / 2:.4f}")
        for (r2, (pq2, tp2, fp2, fn2), _, _) in filas:
            if r2[0] == "votos" and len(jur) == 3 and r2[1]["k"] == 2:
                print(f"  regla 2 de 3, t={r2[1]['t']}: {pq2:.4f} (aciertos {tp2}, inventados {fp2}, escapados {fn2})")
