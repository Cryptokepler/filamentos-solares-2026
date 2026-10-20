"""Detector especialista en filamentos pequeños (ronda 4).

El detector general mira el Sol entero reducido a 1024 px. El especialista mira la imagen
original de 2048 px por trozos ("ventanas") de 1024, así que cada filamento se ve el doble
de grande. Es el mismo Mask R-CNN (detector.construir_detector), arrancando de lo que ya
aprendió el general.

- Estudio: cada vez que ve un ejemplo (imagen x experto) recorta una ventana al azar; casi
  siempre centrada en un filamento pequeño, que es donde falla el general. Ve todos los
  filamentos de la ventana, también trozos de los grandes, para no aprender que eso es fondo.
- Examen y test: 9 ventanas solapadas (pasos de 512). Cada detección se queda solo en la
  ventana que contiene su centro en su zona propia, para no contarla dos veces.
- Juez del especialista: solo deja pasar filamentos de menos de AREA_MAX px (a 2048) y
  descarta lo que ya encontró el detector general (gana el general).
"""
import numpy as np
import torch
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

from . import detector as D

TAM = D.TAM
VENTANA = 1024
PASO = 512
AREA_MAX = 3000


def posiciones(tam=TAM, ventana=VENTANA, paso=PASO):
    return list(range(0, tam - ventana + 1, paso))


def zona_propia(i, tam=TAM, ventana=VENTANA, paso=PASO):
    """[a, b) de la ventana i en un eje: la parte que le toca a ella y no a la vecina."""
    ps = posiciones(tam, ventana, paso)
    borde = (ventana - paso) // 2
    a = 0 if i == 0 else ps[i] + borde
    b = tam if i == len(ps) - 1 else ps[i] + ventana - borde
    return a, b


# ---------------------------------------------------------------- estudio

def _rles_2048(anotaciones):
    return [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], TAM, TAM)) for a in anotaciones]


class DatosEspecialista(torch.utils.data.Dataset):
    """Un ejemplo por (imagen a 2048, anotador); en cada vuelta una ventana distinta.

    p_pequeno: probabilidad de centrar la ventana en un filamento de menos de AREA_MAX px
    (con un desplazamiento al azar); si no, ventana al azar sobre el disco."""

    def __init__(self, ejemplos, aumentar, p_pequeno=0.8):
        self.ejemplos, self.aumentar, self.p_pequeno = ejemplos, aumentar, p_pequeno  # [(img uint8 2048, anotaciones)]

    def __len__(self):
        return len(self.ejemplos)

    def __getitem__(self, i):
        img, anotaciones = self.ejemplos[i]
        rles = _rles_2048(anotaciones)
        areas = mask_utils.area(rles) if rles else np.zeros(0)
        cajas = mask_utils.toBbox(rles) if rles else np.zeros((0, 4))  # x, y, ancho, alto
        pequenos = np.flatnonzero(areas < AREA_MAX)
        tope = TAM - VENTANA
        if len(pequenos) and np.random.rand() < self.p_pequeno:
            x, y, w, h = cajas[np.random.choice(pequenos)]
            cx, cy = x + w / 2 + np.random.uniform(-256, 256), y + h / 2 + np.random.uniform(-256, 256)
        else:
            cx, cy = np.random.uniform(VENTANA / 2, TAM - VENTANA / 2, 2)
        x0 = int(np.clip(round(cx - VENTANA / 2), 0, tope))
        y0 = int(np.clip(round(cy - VENTANA / 2), 0, tope))
        ventana = img[y0:y0 + VENTANA, x0:x0 + VENTANA]
        mascaras = []
        for r, (x, y, w, h) in zip(rles, cajas):
            if x + w <= x0 or x >= x0 + VENTANA or y + h <= y0 or y >= y0 + VENTANA:
                continue
            m = mask_utils.decode(r)[y0:y0 + VENTANA, x0:x0 + VENTANA].astype(bool)
            if m.sum() >= 20:  # trozos minúsculos del borde de la ventana no enseñan nada
                mascaras.append(m)
        return D.objetivo_aumentado(ventana, mascaras, self.aumentar)


# ---------------------------------------------------------------- examen y test

def ventanas(img2048):
    ps = posiciones()
    for iy, y0 in enumerate(ps):
        for ix, x0 in enumerate(ps):
            yield img2048[y0:y0 + VENTANA, x0:x0 + VENTANA], x0, y0, zona_propia(ix), zona_propia(iy)


@torch.no_grad()
def detectar(modelo, img2048, disp, conf_min=0.2):
    """Detecciones pequeñas en toda la imagen: (lista de RLE a 2048, confianzas)."""
    rles, confs = [], []
    for v, x0, y0, (ax, bx), (ay, by) in ventanas(img2048):
        with torch.autocast(disp, enabled=disp == "cuda"):
            m, c = D.detectar(modelo, v, disp)
        for mj, cj in zip(m, c):
            if cj < conf_min:
                continue
            ys, xs = np.nonzero(mj)
            if len(xs) >= AREA_MAX:
                continue
            cx, cy = x0 + xs.mean(), y0 + ys.mean()
            if not (ax <= cx < bx and ay <= cy < by):
                continue  # este filamento le toca a la ventana vecina
            completa = np.zeros((TAM, TAM), np.uint8)
            completa[y0:y0 + VENTANA, x0:x0 + VENTANA] = mj
            rles.append(D.a_rle(completa))
            confs.append(float(cj))
    return rles, np.asarray(confs, np.float32)


def juez(rles, conf, r2048, generales, umbral=0.6, fusionar=True, area_min=0, radio_max=0.95,
         solape_max=0.2):
    """Detecciones del especialista -> máscaras a 2048 que se añaden a las del detector general.

    generales: máscaras a 2048 que ya dio el detector general (tras su juez).
    solape_max: si más de esta fracción del filamento cae sobre uno del general, se descarta
        (gana el general); si es menos, se le quitan esos píxeles.
    """
    elegidos = [r for r, c in zip(rles, conf) if c >= umbral]  # ya vienen de mayor a menor confianza
    if not elegidos:
        return []
    # Se descomprimen de una en una: a 2048 px, cientos de máscaras juntas no caben en memoria
    if fusionar:
        toca = np.asarray(mask_utils.iou(elegidos, elegidos, [0] * len(elegidos))) > 0
        _, etiqueta = connected_components(toca, directed=False)
        mascaras = (mask_utils.decode(mask_utils.merge([elegidos[i] for i in np.flatnonzero(etiqueta == k)],
                                                       intersect=0)).astype(bool)
                    for k in range(etiqueta.max() + 1))
    else:
        mascaras = (mask_utils.decode(r).astype(bool) for r in elegidos)
    del_general = np.logical_or.reduce(generales) if len(generales) else np.zeros((TAM, TAM), bool)
    tomado = np.zeros((TAM, TAM), bool)  # píxeles ya dados a otra detección del especialista
    out = []
    for m in mascaras:
        m = m & ~tomado  # un píxel solo puede ser de un filamento: gana el de más confianza
        area = m.sum()
        if not area_min <= area < AREA_MAX or r2048[m].mean() > radio_max:
            continue
        if (m & del_general).sum() > solape_max * area:
            continue  # ya lo encontró el general
        m = m & ~del_general
        tomado |= m
        out.append(m)
    return out
