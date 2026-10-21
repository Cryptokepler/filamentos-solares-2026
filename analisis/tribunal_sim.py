"""Paso 2 del tribunal: juez B ("la ficha") y simulación de la nota del examen, sin GPU.

Lee los candidatos generosos guardados en el paso 1 (notebook guardar_candidatos) y:
1. Etiqueta cada candidato: con cuántos expertos de su imagen coincide (IoU > 0,5).
2. Le calcula rasgos de contexto (vecinos, solapes, cuántos candidatos hay en la imagen).
3. Entrena el juez B con las imágenes de estudio y lo mide en las del examen.
4. Simula la nota PQ del examen con cada receta de triaje + tribunal.

Uso:
    python3 analisis/tribunal_sim.py <carpeta MAGFiLO_1.0_Kaggle_2026> <carpeta con candidatos_*.json>
"""
import json
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from filamentos.candidatos import de_texto  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

TAM = 2048
RASGOS = ["conf", "area", "area_ref", "radio", "osc_med", "osc_p5", "anillo", "contraste_local", "alarg",
          "ancho", "alto", "rango_conf", "n_cands", "n_fuertes", "solape_fuerte", "dist_fuerte",
          "vecinos_150", "conf_max_vecino"]


def reales_rle(anot_imagen):
    """{experto: [RLE]} a partir de los polígonos (sin pasar por máscaras)."""
    out = []
    for anns in anot_imagen.values():
        out.append([mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], TAM, TAM)) for a in anns])
    return out


def iou(a, b):
    if not a or not b:
        return np.zeros((len(a), len(b)))
    return np.asarray(mask_utils.iou(a, b, [0] * len(b)))


def preparar_imagen(cands, expertos, clave="rle_ref"):
    """Quita repetidos, etiqueta y añade rasgos de contexto. Devuelve (filas, rles)."""
    orden = sorted(range(len(cands)), key=lambda i: -cands[i]["conf"])
    cands = [cands[i] for i in orden]
    rles = [de_texto(c[clave]) for c in cands]
    m = iou(rles, rles)
    dup = np.triu(m > 0.5, 1).any(0)
    cands = [c for c, d in zip(cands, dup) if not d]
    rles = [r for r, d in zip(rles, dup) if not d]
    n = len(cands)
    if not n:
        return [], []
    area = np.array(mask_utils.area(rles), float)
    inter = iou(rles, rles)  # IoU; para la fracción cubierta usamos la intersección
    # intersección = iou * (a + b) / (1 + iou)
    suma = area[:, None] + area[None, :]
    inter = inter * suma / (1 + inter)
    conf = np.array([c["conf"] for c in cands])
    cx = np.array([c["cx"] for c in cands])
    cy = np.array([c["cy"] for c in cands])
    dist = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    fuerte = conf >= 0.8
    aciertos = np.zeros(n)
    for reales in expertos:
        if reales:
            aciertos += iou(rles, reales).max(1) > 0.5
    filas = []
    for i, c in enumerate(cands):
        f = {k: c[k] for k in ("conf", "area", "area_ref", "radio", "osc_med", "osc_p5", "anillo",
                               "contraste_local", "alarg", "ancho", "alto", "cx", "cy")}
        otros_fuertes = fuerte.copy()
        otros_fuertes[i] = False
        f["rango_conf"] = i
        f["n_cands"] = n
        f["n_fuertes"] = int(fuerte.sum())
        f["solape_fuerte"] = float(inter[i, otros_fuertes].max() / area[i]) if otros_fuertes.any() else 0.0
        f["dist_fuerte"] = float(dist[i, otros_fuertes].min()) if otros_fuertes.any() else 3000.0
        vec = (dist[i] < 150) & (np.arange(n) != i)
        f["vecinos_150"] = int(vec.sum())
        f["conf_max_vecino"] = float(conf[vec].max()) if vec.any() else 0.0
        f["aciertos"] = int(aciertos[i])
        f["expertos"] = len(expertos)
        filas.append(f)
    return filas, rles


def cargar(raiz, carpeta, nombre):
    with open(f"{carpeta}/candidatos_{nombre}.json") as fh:
        cands = json.load(fh)
    anot = cargar_anotaciones(raiz)
    datos = {}
    for img, cs in cands.items():
        exps = reales_rle(anot[img])
        filas, rles = preparar_imagen(cs, exps)
        datos[img] = dict(filas=filas, rles=rles, expertos=exps)
    return datos


def montar(seleccion, rles, cands, modo="unir", radio_max=0.95):
    """Candidatos aprobados -> filamentos finales.

    unir: los que se tocan se juntan en uno (como la ronda 3).
    prioridad: se recorren de mayor a menor nota; uno que pisa más de la mitad de su área a uno ya
        aceptado se descarta, y si pisa menos se le recorta lo pisado.
    """
    idx = [i for i in seleccion if cands[i]["radio"] <= radio_max]
    if not idx:
        return []
    sel = [rles[i] for i in idx]
    if modo == "unir":
        toca = iou(sel, sel) > 0
        _, et = connected_components(toca, directed=False)
        return [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
    out = []
    for r in sel:
        if out:
            ocup = mask_utils.merge(out) if len(out) > 1 else out[0]
            a = mask_utils.area(r)
            pis = mask_utils.area(mask_utils.merge([r, ocup], intersect=True))
            if pis > 0.5 * a:
                continue
            if pis > 0:
                m = mask_utils.decode(r).astype(bool) & ~mask_utils.decode(ocup).astype(bool)
                r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
        out.append(r)
    return out


def nota(datos, elegir, modo="unir", imagenes=None):
    """elegir(filas) -> índices aprobados, ordenados de más a menos nota. Devuelve (PQ, SQ, TP, FP, FN)."""
    tp = fp = fn = 0
    suma = 0.0
    for img in imagenes or datos:
        d = datos[img]
        preds = montar(elegir(d["filas"]), d["rles"], d["filas"], modo) if d["filas"] else []
        for reales in d["expertos"]:
            if not preds or not reales:
                fp += len(preds)
                fn += len(reales)
                continue
            m = iou(preds, reales)
            hit = m > 0.5
            tp_i = int(hit.any(0).sum())
            tp += tp_i
            suma += float(m[hit].sum())
            fn += len(reales) - tp_i
            fp += len(preds) - int(hit.any(1).sum())
    den = tp + 0.5 * fp + 0.5 * fn
    sq = suma / tp if tp else 0
    return (suma / den if den else 0), sq, tp, fp, fn


def tabla(datos):
    """Filas por (candidato, experto): X, y (acierta con ese experto), grupo (imagen)."""
    X, y, g, w = [], [], [], []
    for k, (img, d) in enumerate(datos.items()):
        for f in d["filas"]:
            X.append([f[r] for r in RASGOS])
            y.append(f["aciertos"] / f["expertos"])
            g.append(k)
            w.append(f["expertos"])
    return np.array(X, float), np.array(y), np.array(g), np.array(w, float)


if __name__ == "__main__":
    import pickle
    import os

    raiz, carpeta = sys.argv[1], sys.argv[2]
    cache = f"{carpeta}/tribunal_cache.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            exa, est = pickle.load(fh)
    else:
        exa = cargar(raiz, carpeta, "examen")
        est = cargar(raiz, carpeta, "estudio")
        with open(cache, "wb") as fh:
            pickle.dump((exa, est), fh)
    print("candidatos sin repetir: examen", sum(len(d["filas"]) for d in exa.values()),
          "| estudio", sum(len(d["filas"]) for d in est.values()))
    for u in (0.7, 0.75, 0.8, 0.85, 0.9):
        for modo in ("unir", "prioridad"):
            r = nota(exa, lambda fs: [i for i, f in enumerate(fs) if f["conf"] >= u], modo)
            print(f"solo confianza >= {u} ({modo}): PQ {r[0]:.4f} SQ {r[1]:.3f} TP {r[2]} FP {r[3]} FN {r[4]}")
