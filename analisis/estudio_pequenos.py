"""¿Ayudaría mirar más fino (YOLO a 1536 o modelo más grande)? Estudio sin GPU de los escapados (22 oct 2026).

Uso:
    python3 analisis/estudio_pequenos.py <carpeta MAGFiLO_1.0_Kaggle_2026> <candidatos_yolo_examen.json>

Para cada filamento de experto que se escapa con YOLO + refinador (umbral 0,25):
- ¿YOLO lo vio flojito (algún candidato con IoU > 0,5 y confianza < 0,25) o no lo vio nada?
- ¿Otro experto de la misma imagen lo dibujó también (confirmado)?
- Su grosor a 1024 px (lo que ve YOLO): si es de 1-3 px, a 1536 se vería mejor.
Y cuánto subiría la nota si se recuperaran, sin inventar nada más.
"""
import json
import sys
from collections import Counter

import cv2
import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.techo_borde import pq, seleccion  # noqa: E402
from analisis.tribunal_sim import iou, reales_rle  # noqa: E402
from analisis.votacion_r5 import limpiar  # noqa: E402
from filamentos.datos import cargar_anotaciones  # noqa: E402

UMBRAL = 0.25


def grosor(rle):
    """Grosor máximo en píxeles a 2048 (doble del máximo de la transformada de distancia)."""
    x, y, w, h = (int(v) for v in mask_utils.toBbox(rle))
    m = mask_utils.decode(rle)[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2]
    return 2 * float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max())


def tamano(a):
    return "<1k" if a < 1000 else "1k-3k" if a < 3000 else ">3k"


if __name__ == "__main__":
    raiz, fcand = sys.argv[1:3]
    anot = cargar_anotaciones(raiz)
    cand = json.load(open(fcand))
    imgs = sorted(cand)
    reales = {n: reales_rle(anot[n]) for n in imgs}
    sel = {n: seleccion(cand[n], UMBRAL) for n in imgs}
    base = pq(sel, reales)
    print(f"Base: PQ {base[0]:.4f} SQ {base[1]:.3f} aciertos {base[2]} inventados {base[3]} escapados {base[4]}")

    filas = []
    for n in imgs:
        conf, todos = limpiar(cand[n])
        for e, rs in enumerate(reales[n]):
            if not rs:
                continue
            hit = iou(sel[n], rs).max(0) > 0.5 if sel[n] else np.zeros(len(rs), bool)
            vis = iou(todos, rs) if todos else np.zeros((0, len(rs)))
            otros = [r for k, r in enumerate(reales[n]) if k != e]
            for j, r in enumerate(rs):
                if hit[j]:
                    continue
                visto = vis.shape[0] and (vis[:, j] > 0.5).any()
                rozado = vis.shape[0] and (vis[:, j] > 0.1).any()
                conf_otros = [bool(o) and iou(o, [r]).max() > 0.5 for o in otros]
                area = float(mask_utils.area(r))
                filas.append(dict(tam=tamano(area), visto="flojito" if visto else "rozado" if rozado else "nada",
                                  multi=len(otros) > 0, confirmado=any(conf_otros),
                                  grosor1024=grosor(r) / 2, iou_mejor=float(vis[:, j].max()) if vis.shape[0] else 0.0))
    print(f"\n{len(filas)} escapados")
    for t in ("<1k", "1k-3k", ">3k"):
        f = [x for x in filas if x["tam"] == t]
        c = Counter(x["visto"] for x in f)
        g = np.array([x["grosor1024"] for x in f])
        conf = [x for x in f if x["multi"]]
        print(f"  {t:6s} {len(f):4d} | YOLO lo vio flojito {c['flojito']:3d}, lo rozó mal dibujado {c['rozado']:3d}, "
              f"nada {c['nada']:3d} | grosor a 1024: mediana {np.median(g):.1f} px, <=3 px {int((g <= 3).sum())} | "
              f"en imágenes con varios expertos {len(conf)}, confirmados por otro {sum(x['confirmado'] for x in conf)}")
    nada = [x for x in filas if x["visto"] == "nada"]
    print(f"\nNo vistos en absoluto: {len(nada)}; de ellos finos (<=3 px a 1024): "
          f"{sum(x['grosor1024'] <= 3 for x in nada)}; confirmados por otro experto: {sum(x['confirmado'] for x in nada)} "
          f"(de {sum(x['multi'] for x in nada)} en imágenes con varios expertos)")

    # Techo: recuperar escapados sin inventar nada. Cada uno suma 1 acierto (con SQ medio) y quita 1 escapado.
    p, sq, tp, fp, fn = base[:5]
    suma = p * (tp + 0.5 * fp + 0.5 * fn)
    for nombre, k in (("todos los no vistos", len(nada)),
                      ("no vistos y finos", sum(x["grosor1024"] <= 3 for x in nada)),
                      ("no vistos pequeños (<3k)", sum(x["tam"] != ">3k" for x in nada))):
        nuevo = (suma + k * sq) / (tp + k + 0.5 * fp + 0.5 * (fn - k))
        print(f"  Techo si recuperamos {nombre} ({k}) sin inventar nada: PQ {nuevo:.4f} (+{nuevo - p:.4f})")
    # Realista: con la puntería de la ronda 4 (recuperó 88, inventó ~170)
    for k in (40, 88):
        nuevo = (suma + k * sq) / (tp + k + 0.5 * (fp + 2 * k) + 0.5 * (fn - k))
        print(f"  Si recupera {k} e inventa {2 * k} (puntería de la ronda 4): PQ {nuevo:.4f} ({nuevo - p:+.4f})")
    k = 88
    for inv in (44, 88):
        nuevo = (suma + k * sq) / (tp + k + 0.5 * (fp + inv) + 0.5 * (fn - k))
        print(f"  Si recupera {k} e inventa {inv}: PQ {nuevo:.4f} ({nuevo - p:+.4f})")
