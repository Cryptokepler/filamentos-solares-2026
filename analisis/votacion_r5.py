"""Votación sin GPU: candidatos de la ronda 3 (Mask R-CNN) + ronda 5 (YOLO11-seg) en el examen.

Uso:
    python3 analisis/votacion_r5.py <carpeta MAGFiLO_1.0_Kaggle_2026> <candidatos_examen.json ronda 3> <candidatos_yolo_examen.json>

Cada candidato lleva su silueta refinada (rle_ref) y su confianza. Se emparejan los de los dos modelos
(IoU > 0,5) y se prueban recetas; la elegida se mide de forma honesta: umbral elegido en una mitad del
examen y medido en la otra.
"""
import itertools
import json
import sys

import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from filamentos.candidatos import de_texto  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

RADIO_MAX = 0.95


def limpiar(cands):
    """Ordena por confianza, quita repetidos (IoU > 0,5) y los del borde. Devuelve (confs, rles)."""
    cands = sorted((c for c in cands if c["radio"] <= RADIO_MAX), key=lambda c: -c["conf"])
    rles = [de_texto(c["rle_ref"]) for c in cands]
    if not rles:
        return np.zeros(0), []
    dup = np.triu(iou(rles, rles) > 0.5, 1).any(0)
    return np.array([c["conf"] for c, d in zip(cands, dup) if not d]), [r for r, d in zip(rles, dup) if not d]


def preparar(c3, c5):
    """Por imagen: candidatos de cada modelo y su pareja en el otro (índice o -1)."""
    out = {}
    for n in c3:
        ca, ra = limpiar(c3[n])
        cb, rb = limpiar(c5.get(n, []))
        m = iou(ra, rb)
        pa = np.full(len(ra), -1)
        pb = np.full(len(rb), -1)
        if m.size:
            for i, j in sorted(zip(*np.nonzero(m > 0.5)), key=lambda x: -m[x]):
                if pa[i] < 0 and pb[j] < 0:
                    pa[i], pb[j] = j, i
        out[n] = dict(ca=ca, ra=ra, cb=cb, rb=rb, pa=pa, pb=pb)
    return out


def elegir(d, receta):
    """Lista de RLE elegidas en una imagen según la receta (tipo, parámetros)."""
    tipo, p = receta
    ca, cb, pa, pb = d["ca"], d["cb"], d["pa"], d["pb"]
    sel = []  # (puntuación, rle)
    if tipo == "solo_r3":
        sel = [(c, r) for c, r in zip(ca, d["ra"]) if c >= p["ta"]]
    elif tipo == "solo_yolo":
        sel = [(c, r) for c, r in zip(cb, d["rb"]) if c >= p["tb"]]
    elif tipo == "media":
        # Puntuación combinada: w * conf ronda 3 + (1 - w) * conf YOLO (0 si el otro no lo ve)
        w = p["w"]
        for i, j in enumerate(pa):
            sc = w * ca[i] + (1 - w) * (cb[j] if j >= 0 else 0)
            if sc >= p["t"]:
                sel.append((sc, d["rb"][j] if j >= 0 and p.get("silueta") == "yolo" else d["ra"][i]))
        sel += [((1 - w) * c, r) for c, r, i in zip(cb, d["rb"], pb) if i < 0 and (1 - w) * c >= p["t"]]
    elif tipo in ("union", "acuerdo"):
        # Parejas: entran si cualquiera (union) o con umbrales de acuerdo (acuerdo) ; silueta del más seguro
        for i, j in enumerate(pa):
            if j >= 0:
                ok = (ca[i] >= p["ta"] or cb[j] >= p["tb"]) if tipo == "union" else \
                     (ca[i] >= p["pa_min"] and cb[j] >= p["pb_min"]) or ca[i] >= p["ta"] or cb[j] >= p["tb"]
                if ok:
                    sel.append((max(ca[i], cb[j]) + 1, d["ra"][i] if p.get("silueta") == "r3" or
                                (p.get("silueta") != "yolo" and ca[i] >= cb[j]) else d["rb"][j]))
        sel += [(c + 0.5, r) for c, r, j in zip(ca, d["ra"], pa) if j < 0 and c >= p["ta"]]
        sel += [(c, r) for c, r, i in zip(cb, d["rb"], pb) if i < 0 and c >= p["tb"]]
    sel.sort(key=lambda x: -x[0])
    rles = [r for _, r in sel]
    if p.get("no_tocar") and len(rles) > 1:
        # un candidato que toca a uno ya aceptado (de más puntuación) se descarta en vez de unirse
        acept = []
        for r in rles:
            if not acept or not (iou([r], acept) > 0).any():
                acept.append(r)
        return acept
    return rles


def nota(datos, reales, receta, imagenes=None):
    tp = fp = fn = 0
    suma = 0.0
    for n in imagenes or datos:
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


def recetas():
    for w, t, nt in itertools.product((0.2, 0.35, 0.5, 0.65), (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6),
                                     (False, True)):
        yield ("media", dict(w=w, t=t, no_tocar=nt, silueta="yolo"))
    for ta in (0.7, 0.8, 0.9):
        yield ("solo_r3", dict(ta=ta))
    for tb in (0.15, 0.2, 0.25, 0.3):
        yield ("solo_yolo", dict(tb=tb))
    for ta, tb, nt, sil in itertools.product((0.8, 0.9, 0.95, 1.1), (0.2, 0.25, 0.3, 0.4, 0.5),
                                            (False, True), ("mejor", "r3", "yolo")):
        yield ("union", dict(ta=ta, tb=tb, no_tocar=nt, silueta=sil))
    for pmin, ta, tb, nt in itertools.product(((0.3, 0.1), (0.5, 0.15), (0.5, 0.2), (0.3, 0.2), (0.6, 0.1)),
                                              (0.8, 0.9, 1.1), (0.3, 0.4, 0.5, 1.1), (False, True)):
        yield ("acuerdo", dict(pa_min=pmin[0], pb_min=pmin[1], ta=ta, tb=tb, no_tocar=nt, silueta="mejor"))


if __name__ == "__main__":
    raiz, f3, f5 = sys.argv[1:4]
    c3, c5 = json.load(open(f3)), json.load(open(f5))
    anot = cargar_anotaciones(raiz)
    imgs = sorted(set(c3) & set(c5))
    print(f"{len(imgs)} imágenes del examen (ronda 3: {len(c3)}, YOLO: {len(c5)})")
    reales = {n: reales_rle(anot[n]) for n in imgs}
    datos = preparar({n: c3[n] for n in imgs}, c5)
    np_a = sum(len(d["ca"]) for d in datos.values())
    np_b = sum(len(d["cb"]) for d in datos.values())
    par = sum((d["pa"] >= 0).sum() for d in datos.values())
    print(f"candidatos: ronda 3 {np_a}, YOLO {np_b}, emparejados {par}")
    rng = np.random.default_rng(0)
    orden = list(imgs)
    rng.shuffle(orden)
    mA, mB = orden[:60], orden[60:]
    filas = []
    for r in recetas():
        filas.append((r, nota(datos, reales, r), nota(datos, reales, r, mA)[0], nota(datos, reales, r, mB)[0]))
    filas.sort(key=lambda x: -x[1][0])
    print("\nMejores 15 recetas (examen completo):")
    for r, (pq, tp, fp, fn), a, b in filas[:15]:
        print(f"  {pq:.4f} TP {tp} FP {fp} FN {fn} | mitades {a:.4f} / {b:.4f} | {r}")
    for tipo in ("solo_r3", "solo_yolo", "union", "acuerdo", "media"):
        f = [x for x in filas if x[0][0] == tipo]
        ra = max(f, key=lambda x: x[2])
        rb = max(f, key=lambda x: x[3])
        best = f[0]
        print(f"\n{tipo}: mejor {best[1][0]:.4f} (TP {best[1][1]} FP {best[1][2]} FN {best[1][3]}) {best[0][1]}")
        print(f"  honesto: elegida en mitad A -> en B {ra[3]:.4f}; elegida en B -> en A {rb[2]:.4f}; "
              f"media {(ra[3] + rb[2]) / 2:.4f}")
