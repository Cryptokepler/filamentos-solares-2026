"""Paso 1 del tribunal: el detector en modo generoso y la ficha de cada candidato.

El detector de la ronda 3 propone todo lo que ve con confianza >= conf_min (en vez de 0,8). De cada
candidato se guarda su silueta (la del detector y la del refinador, cada una por separado, sin unir
solapes), una ficha de rasgos y, si se pide, un recorte para el juez A. Con eso se ajustan las perillas
del tribunal fuera de Kaggle, sin GPU (docs/arquitectura-tribunal.md).
"""
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

from . import clasico
from . import detector as D

TAM = D.TAM
FRANJAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01)


def radio(img):
    """Distancia al centro del disco solar, en radios, para cada píxel."""
    cx, cy, R = clasico.disco_solar(img.astype(np.float32))
    yy, xx = np.indices(img.shape)
    return (np.hypot(xx - cx, yy - cy) / R).astype(np.float32)


def a_texto(m):
    r = D.a_rle(m)
    return r["counts"].decode("ascii")


def de_texto(counts):
    return {"size": [TAM, TAM], "counts": counts.encode("ascii")}


def _ficha(m, rel, r2048, conf, m_res):
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(ys.min() - 12, 0), ys.max() + 13, max(xs.min() - 12, 0), xs.max() + 13
    trozo = m[y0:y1, x0:x1]
    anillo = ndi.binary_dilation(trozo, iterations=10) & ~ndi.binary_dilation(trozo, iterations=2)
    dentro = rel[y0:y1, x0:x1][trozo]
    fuera = rel[y0:y1, x0:x1][anillo]
    med_fuera = float(np.median(fuera)) if fuera.size else 1.0
    return dict(conf=round(float(conf), 4), area=int(m.sum()), cx=round(float(xs.mean()), 1),
                cy=round(float(ys.mean()), 1), ancho=int(x1 - x0), alto=int(y1 - y0),
                radio=round(float(r2048[m].mean()), 4), osc_med=round(float(np.median(dentro)), 4),
                osc_p5=round(float(np.percentile(dentro, 5)), 4), anillo=round(med_fuera, 4),
                contraste_local=round(float(np.median(dentro)) / max(med_fuera, 1e-3), 4),
                alarg=round(D.alargamiento(m_res), 3))


def procesar(gen, ref, grande, disp, res=1024, conf_min=0.1, umbral_ref=0.5, lado_recorte=0, detectar=None):
    """Una imagen a 2048 -> (candidatos, recortes, predicción de la ronda 3 para comprobar).

    detectar: función img[res,res] -> (máscaras [N,res,res], confianzas [N]) para usar otro detector
    (por ejemplo YOLO); por defecto, el Mask R-CNN gen.

    candidatos: lista de dict con la ficha, 'rle' (detector) y 'rle_ref' (refinador).
    recortes: uint8 [N, 2, lado, lado] (imagen y silueta) o None si lado_recorte=0.
    """
    import torch

    img = np.asarray(Image.fromarray(grande).resize((res, res), Image.BILINEAR), np.uint8)
    r = radio(img)
    r2048 = np.asarray(Image.fromarray(r).resize((TAM, TAM), Image.BILINEAR))
    rel = D.brillo_relativo(grande, r2048).astype(np.float32)
    with torch.autocast(disp, enabled=disp == "cuda"):
        m, c = detectar(img) if detectar else D.detectar(gen, img, disp)
    # Lo mismo que la ronda 3 (juez 0,8 uniendo solapes + refinador 0,5), para comprobar que todo cuadra
    r3 = D.instancias(m, c, r, umbral=0.8, fusionar=True, area_min=0)
    with torch.autocast(disp, enabled=disp == "cuda"):
        r3 = D.refinar(ref, grande, r3, disp, umbral=umbral_ref)
    keep = c >= conf_min
    m, c = m[keep], c[keep]
    m2048 = [D.a_2048(x) for x in m]
    keep = [i for i, x in enumerate(m2048) if x.any()]
    m, c, m2048 = m[keep], c[keep], [m2048[i] for i in keep]
    with torch.autocast(disp, enabled=disp == "cuda"):
        refinadas = D.refinar(ref, grande, m2048, disp, umbral=umbral_ref, unir=False)
    cands, recs = [], []
    for mi, ci, mg, mr in zip(m, c, m2048, refinadas):
        f = _ficha(mg, rel, r2048, ci, mi)
        f["area_ref"] = int(mr.sum())
        f["rle"], f["rle_ref"] = a_texto(mg), a_texto(mr)
        cands.append(f)
        if lado_recorte:
            caja = D.caja_cuadrada(mg)
            x0, y0, x1, y1 = caja
            a = np.asarray(Image.fromarray(grande[y0:y1, x0:x1]).resize((lado_recorte,) * 2, Image.BILINEAR))
            b = np.asarray(Image.fromarray(mg[y0:y1, x0:x1].astype(np.uint8) * 255).resize(
                (lado_recorte,) * 2, Image.NEAREST))
            recs.append(np.stack([a, b]))
    recortes = np.stack(recs).astype(np.uint8) if recs else (
        np.zeros((0, 2, lado_recorte, lado_recorte), np.uint8) if lado_recorte else None)
    return cands, recortes, r3


def duplicados(cands, clave="rle_ref", umbral=0.5):
    """True para los candidatos que repiten (IoU > umbral) a otro de más confianza."""
    if not cands:
        return np.zeros(0, bool)
    orden = np.argsort([-x["conf"] for x in cands], kind="stable")
    rles = [de_texto(cands[i][clave]) for i in orden]
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
    dup_ord = np.triu(iou > umbral, 1).any(0)
    dup = np.zeros(len(cands), bool)
    dup[orden] = dup_ord
    return dup


def tabla_franjas(cands_por_imagen, reales_por_imagen, clave="rle_ref"):
    """Acierto por franja de confianza y cuántos filamentos reales quedan al alcance.

    cands_por_imagen: {imagen: [candidato]}; reales_por_imagen: {imagen: [[RLE de un experto], ...]}.
    Cada candidato se compara con cada experto por separado, como la métrica. Los candidatos que
    repiten a otro de más confianza (IoU > 0,5) no cuentan: el tribunal se quedaría con uno.
    Devuelve (filas por franja, reales totales, reales que algún candidato alcanza con IoU > 0,5,
    duplicados descartados).
    """
    franjas = np.asarray(FRANJAS)
    aciertos = np.zeros(len(franjas) - 1)
    total = np.zeros(len(franjas) - 1)
    suma_iou = np.zeros(len(franjas) - 1)
    n_reales = alcanzables = n_dup = 0
    for n, cands in cands_por_imagen.items():
        dup = duplicados(cands, clave)
        n_dup += int(dup.sum())
        cands = [x for x, d in zip(cands, dup) if not d]
        rles = [de_texto(x[clave]) for x in cands]
        conf = np.array([x["conf"] for x in cands])
        k = np.clip(np.searchsorted(franjas, conf, side="right") - 1, 0, len(franjas) - 2)
        for reales in reales_por_imagen[n]:
            n_reales += len(reales)
            if not rles:
                continue
            if reales:
                iou = np.asarray(mask_utils.iou(rles, reales, [0] * len(reales)))
                mejor = iou.max(1)
                alcanzables += int((iou.max(0) > 0.5).sum())
            else:
                mejor = np.zeros(len(rles))
            np.add.at(total, k, 1)
            np.add.at(aciertos, k, mejor > 0.5)
            np.add.at(suma_iou, k, np.where(mejor > 0.5, mejor, 0))
    filas = []
    for i in range(len(franjas) - 1):
        if total[i]:
            filas.append(dict(franja=f"{franjas[i]:.2f}-{min(franjas[i + 1], 1):.2f}", candidatos=int(total[i]),
                              aciertos=int(aciertos[i]), pct_acierto=round(float(100 * aciertos[i] / total[i]), 1),
                              iou_medio=round(float(suma_iou[i] / max(aciertos[i], 1)), 3)))
    return filas, n_reales, alcanzables, n_dup
