"""Observabilidad: ¿fallan los dos detectores (ronda 3 y YOLO) en los mismos filamentos? ¿Y los expertos?

Uso:
    python3 analisis/observar_fallos.py <carpeta MAGFiLO> <candidatos_examen.json ronda 3> <candidatos_yolo_examen.json>
Para cada filamento real del examen (por experto) dice si lo acierta cada modelo y cuántos de los otros
expertos de la imagen dibujaron lo mismo (IoU > 0,5).
"""
import json
import sys
from collections import Counter

import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import elegir, preparar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402


def finales(d, receta):
    sel = elegir(d, receta)
    if len(sel) > 1:
        _, et = connected_components(iou(sel, sel) > 0, directed=False)
        sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
    return sel


def tam(a):
    return "<1k" if a < 1000 else "1k-3k" if a < 3000 else ">3k"


if __name__ == "__main__":
    raiz, f3, f5 = sys.argv[1:4]
    c3, c5 = json.load(open(f3)), json.load(open(f5))
    anot = cargar_anotaciones(raiz)
    imgs = sorted(set(c3) & set(c5))
    datos = preparar({n: c3[n] for n in imgs}, c5)
    filas = []
    for n in imgs:
        expertos = reales_rle(anot[n])
        sa = finales(datos[n], ("solo_r3", dict(ta=0.8)))
        sb = finales(datos[n], ("solo_yolo", dict(tb=0.25)))
        for e, rs in enumerate(expertos):
            if not rs:
                continue
            ha = (iou(rs, sa) > 0.5).any(1) if sa else np.zeros(len(rs), bool)
            hb = (iou(rs, sb) > 0.5).any(1) if sb else np.zeros(len(rs), bool)
            otros = [o for k, o in enumerate(expertos) if k != e]
            coinc = sum((iou(rs, o) > 0.5).any(1) for o in otros if o) if otros else None
            for k, a in enumerate(mask_utils.area(rs)):
                filas.append(dict(img=n, a=bool(ha[k]), b=bool(hb[k]), area=int(a), n_otros=len(otros),
                                  coinc=None if coinc is None or isinstance(coinc, int) else int(coinc[k])))
    print(f"filamentos reales (imagen, experto): {len(filas)}")
    cat = Counter(("los dos" if f["a"] and f["b"] else "solo r3" if f["a"] else "solo YOLO" if f["b"] else "ninguno")
                  for f in filas)
    print("quién acierta:", dict(cat))
    print("\nescapados de los dos, por tamaño:", dict(Counter(tam(f["area"]) for f in filas if not f["a"] and not f["b"])))
    print("\nen imágenes con varios expertos: % de filamentos que otro experto también dibujó")
    for nombre, cond in (("los aciertan los dos", lambda f: f["a"] and f["b"]),
                         ("solo uno de los modelos", lambda f: f["a"] != f["b"]),
                         ("no los ve ninguno", lambda f: not f["a"] and not f["b"])):
        g = [f for f in filas if f["n_otros"] and cond(f)]
        if g:
            alguno = np.mean([f["coinc"] > 0 for f in g])
            print(f"  {nombre}: {len(g)} filamentos, otro experto coincide en el {100 * alguno:.0f} %")
            for t in ("<1k", "1k-3k", ">3k"):
                gt = [f for f in g if tam(f["area"]) == t]
                if gt:
                    print(f"      {t}: {len(gt)}, coincide otro experto {100 * np.mean([f['coinc'] > 0 for f in gt]):.0f} %")
