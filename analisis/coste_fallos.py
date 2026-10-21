"""¿Qué nos cuesta más puntos? Desglose de inventados y escapados del mejor sistema (YOLO + refinador).

Uso: python3 analisis/coste_fallos.py <carpeta MAGFiLO> <candidatos_examen.json r3> <candidatos_yolo_examen.json> [umbral]
Cada inventado se clasifica: "otro experto lo dibujó" (en la misma imagen otro experto tiene ese filamento),
"casi" (toca un real con IoU 0,2-0,5: mal partido/unido/dibujado) o "inventado puro".
"""
import json
import sys
from collections import Counter

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.observar_fallos import finales  # noqa: E402
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import preparar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

if __name__ == "__main__":
    raiz, f3, f5 = sys.argv[1:4]
    umbral = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    c3, c5 = json.load(open(f3)), json.load(open(f5))
    anot = cargar_anotaciones(raiz)
    imgs = sorted(set(c3) & set(c5))
    datos = preparar({n: c3[n] for n in imgs}, c5)
    tp = 0
    suma = 0.0
    fp, fn = Counter(), Counter()
    for n in imgs:
        expertos = reales_rle(anot[n])
        sel = finales(datos[n], ("solo_yolo", dict(tb=umbral)))
        varios = sum(1 for e in expertos if e) > 1
        for e, rs in enumerate(expertos):
            otros = [r for k, o in enumerate(expertos) if k != e for r in o]
            m = iou(sel, rs) if sel and rs else np.zeros((len(sel), len(rs)))
            hit = m > 0.5
            tp += int(hit.any(0).sum())
            suma += float(m[hit].sum())
            for i in np.flatnonzero(~hit.any(1)) if len(sel) else []:
                if otros and (iou([sel[i]], otros) > 0.5).any():
                    fp["otro experto lo dibujó"] += 1
                elif m.shape[1] and m[i].max() > 0.2:
                    fp["casi (partido/unido/mal dibujado)"] += 1
                else:
                    fp["inventado puro" + ("" if varios else " (imagen de 1 experto)")] += 1
            for j in np.flatnonzero(~hit.any(0)) if len(rs) else []:
                if not varios:
                    fn["imagen de 1 experto"] += 1
                elif (iou([rs[j]], otros) > 0.5).any():
                    fn["otro experto lo confirma"] += 1
                else:
                    fn["solo lo dibujó este experto"] += 1
    nfp, nfn = sum(fp.values()), sum(fn.values())
    den = tp + 0.5 * nfp + 0.5 * nfn
    pq = suma / den
    print(f"umbral {umbral}: PQ {pq:.4f} SQ {suma / tp:.3f} TP {tp} FP {nfp} FN {nfn}")
    print(f"techo si los bordes fueran perfectos (SQ=1): {tp / den:.4f}")
    for nombre, c in (("INVENTADOS", fp), ("ESCAPADOS", fn)):
        print(f"\n{nombre}")
        for k, v in c.most_common():
            print(f"  {k}: {v}  -> si desaparecieran, PQ {suma / (den - 0.5 * v):.4f} (+{suma / (den - 0.5 * v) - pq:.3f})")
