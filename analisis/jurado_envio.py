"""submission.csv del jurado de YOLO a partir de los candidatos del test ya guardados (25 oct 2026, sin GPU).

Uso:
    python3 analisis/jurado_envio.py <salida.csv> <t> <k> <candidatos_test_modelo1.json> <candidatos_test_modelo2.json> ...

Misma regla que analisis/jurado_yolo.py: los candidatos de los distintos YOLO que se solapan (IoU > 0,5) forman un
caso; pasa si al menos k modelos lo ven con confianza >= t; su silueta es la del candidato más seguro; después se
unen los elegidos que se tocan (como el juez). Ronda 5 + ronda 7: t=0,2, k=2 (0,4607 honesto en el examen).
"""
import csv
import json
import os
import sys

import numpy as np
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.jurado_yolo import casos, elegir  # noqa: E402
from analisis.tribunal_sim import iou  # noqa: E402
from analisis.votacion_r5 import limpiar  # noqa: E402


def jurado(por_modelo, t, k):
    sel = elegir(casos([limpiar(c) for c in por_modelo]), ("votos", dict(t=t, k=k)))
    if len(sel) > 1:
        _, et = connected_components(iou(sel, sel) > 0, directed=False)
        sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == g)]) for g in range(et.max() + 1)]
    return sel


if __name__ == "__main__":
    salida, t, k = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    modelos = [json.load(open(f)) for f in sys.argv[4:]]
    imgs = sorted(set().union(*modelos))
    filas = []
    for f in imgs:
        img_id = os.path.splitext(f)[0]
        for n, r in enumerate(jurado([m.get(f, []) for m in modelos], t, k), start=1):
            c = r["counts"]
            filas.append((f"{img_id}_{n}", c.decode("ascii") if isinstance(c, bytes) else c))
    with open(salida, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(filas)
    print(len(filas), "filamentos en", len({a.rsplit("_", 1)[0] for a, _ in filas}), "de", len(imgs), "imágenes ->", salida)
