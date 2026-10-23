"""¿Perdemos puntos por partir o unir filamentos? Y ¿ayuda unir trozos cercanos ("gap")? (23 oct 2026, sin GPU)

Uso:
    python3 analisis/partir_unir.py <carpeta MAGFiLO_1.0_Kaggle_2026> <candidatos_yolo_examen.json>

A) Con YOLO + refinador (umbral 0,25), clasifica cada escapado e inventado:
   - partido: un filamento del experto que nuestros trozos cubren juntos (IoU > 0,5) pero ninguno solo;
   - fusionado: una predicción nuestra que junta 2+ filamentos del experto (la unión de ellos IoU > 0,5).
B) Barrido de unión por cercanía: dos predicciones a menos de N píxeles se unen en un solo filamento
   (sin engordar el dibujo), para varios N y umbrales.
"""
import json
import sys

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.techo_borde import morfo, pq  # noqa: E402
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import limpiar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402


def unir(sel, gap):
    """Une las predicciones que se tocan después de engordarlas gap/2 píxeles; devuelve las originales unidas."""
    if len(sel) < 2:
        return sel
    gordas = [morfo(r, (gap + 1) // 2) for r in sel] if gap > 0 else sel
    _, et = connected_components(iou(gordas, gordas) > 0, directed=False)
    return [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]


def clasificar(sel, rs):
    """Cuenta escapados partidos e inventados que fusionan, para una imagen y un experto."""
    if not sel or not rs:
        return 0, 0
    m = iou(sel, rs)
    hit = m > 0.5
    partidos = fusiones = 0
    for j in np.flatnonzero(~hit.any(0)):
        trozos = [sel[i] for i in np.flatnonzero((m[:, j] > 0) & ~hit.any(1))]
        if len(trozos) >= 2 and iou([mask_utils.merge(trozos)], [rs[j]])[0, 0] > 0.5:
            partidos += 1
    for i in np.flatnonzero(~hit.any(1)):
        juntos = [rs[j] for j in np.flatnonzero(m[i] > 0)]
        if len(juntos) >= 2 and iou([sel[i]], [mask_utils.merge(juntos)])[0, 0] > 0.5:
            fusiones += 1
    return partidos, fusiones


if __name__ == "__main__":
    raiz, fcand = sys.argv[1:3]
    anot = cargar_anotaciones(raiz)
    cand = json.load(open(fcand))
    imgs = sorted(cand)
    reales = {n: reales_rle(anot[n]) for n in imgs}
    limpio = {n: limpiar(cand[n]) for n in imgs}

    def sel_de(umbral, gap):
        return {n: unir([r for c, r in zip(*limpio[n]) if c >= umbral], gap) for n in imgs}

    base = sel_de(0.25, 0)
    p, sq, tp, fp, fn, _ = pq(base, reales)
    part = fus = 0
    for n in imgs:
        for rs in reales[n]:
            a, b = clasificar(base[n], rs)
            part += a
            fus += b
    print(f"A) Base (umbral 0,25, unir solo si se tocan): PQ {p:.4f} SQ {sq:.3f} RQ {p / sq:.3f} | "
          f"aciertos {tp} inventados {fp} escapados {fn}")
    print(f"   escapados porque los partimos en trozos: {part} de {fn}; "
          f"inventados porque juntamos varios filamentos en uno: {fus} de {fp}")

    print("\nB) Unir trozos a menos de N píxeles (a 2048)")
    for umbral in (0.2, 0.25, 0.3):
        for gap in (0, 5, 10, 20, 40):
            p, sq, tp, fp, fn, _ = pq(sel_de(umbral, gap), reales)
            print(f"   umbral {umbral} gap {gap:2d}: PQ {p:.4f} SQ {sq:.3f} | aciertos {tp} inventados {fp} escapados {fn}",
                  flush=True)
