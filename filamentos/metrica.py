"""Panoptic Quality (PQ) tal como la define la competición, más diagnósticos.

Un filamento predicho y uno real se emparejan si su IoU > 0.5. Con ese umbral
cada filamento solo puede emparejarse con uno del otro lado, así que no hace
falta un emparejamiento óptimo.

    PQ = (suma de IoU de los aciertos / TP) * (TP / (TP + FP/2 + FN/2))
"""
from dataclasses import dataclass, field

import numpy as np

UMBRAL_IOU = 0.5


@dataclass
class Resultado:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    suma_iou: float = 0.0
    # Diagnósticos que la rúbrica también mira
    uno_a_muchos: int = 0   # un filamento real cubierto por 2+ predichos (fragmentación)
    muchos_a_uno: int = 0   # un predicho que cubre 2+ reales (fusión)
    ious: list = field(default_factory=list)

    @property
    def sq(self):
        return self.suma_iou / self.tp if self.tp else 0.0

    @property
    def rq(self):
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / den if den else 1.0

    @property
    def pq(self):
        return self.sq * self.rq

    def __iadd__(self, otro):
        for k in ("tp", "fp", "fn", "suma_iou", "uno_a_muchos", "muchos_a_uno"):
            setattr(self, k, getattr(self, k) + getattr(otro, k))
        self.ious += otro.ious
        return self


def _matriz_solape(reales, predichas):
    """Intersecciones y áreas usando etiquetas, sin comparar máscara contra máscara."""
    if not reales or not predichas:
        return np.zeros((len(reales), len(predichas))), None, None
    forma = reales[0].shape
    area_r = np.array([m.sum() for m in reales], float)
    area_p = np.array([m.sum() for m in predichas], float)
    inter = np.zeros((len(reales), len(predichas)))
    for i, r in enumerate(reales):
        # Recorte a la caja de r para no recorrer la imagen entera
        ys, xs = np.nonzero(r)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rr = r[y0:y1, x0:x1]
        for j, p in enumerate(predichas):
            inter[i, j] = np.logical_and(rr, p[y0:y1, x0:x1]).sum()
    return inter, area_r, area_p


def evaluar_imagen(reales, predichas):
    """reales, predichas: listas de máscaras booleanas 2D del mismo tamaño."""
    res = Resultado()
    inter, area_r, area_p = _matriz_solape(reales, predichas)
    if area_r is None:
        res.fn, res.fp = len(reales), len(predichas)
        return res
    iou = inter / (area_r[:, None] + area_p[None, :] - inter)
    emparejados_r = set()
    emparejados_p = set()
    for i, j in zip(*np.nonzero(iou > UMBRAL_IOU)):
        emparejados_r.add(i)
        emparejados_p.add(j)
        res.suma_iou += iou[i, j]
        res.ious.append(float(iou[i, j]))
    res.tp = len(emparejados_r)
    res.fn = len(reales) - res.tp
    res.fp = len(predichas) - len(emparejados_p)

    # Un solape cuenta si cubre al menos el 10 % del filamento más pequeño de la pareja
    toca = inter > 0.1 * np.minimum(area_r[:, None], area_p[None, :])
    res.uno_a_muchos = int((toca.sum(axis=1) >= 2).sum())
    res.muchos_a_uno = int((toca.sum(axis=0) >= 2).sum())
    return res


def resumen(res):
    return (f"PQ {res.pq:.4f} (forma SQ {res.sq:.3f} x detección RQ {res.rq:.3f}) | "
            f"TP {res.tp} FP {res.fp} FN {res.fn} | "
            f"fragmentados {res.uno_a_muchos} fusionados {res.muchos_a_uno}")
