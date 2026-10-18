"""Línea base clásica: observador (zonas oscuras) + juez (reglas observadas).

Sin aprendizaje automático. Sirve para estrenar el formato de envío y como
referencia contra la que medir los modelos.
"""
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology


@dataclass
class Parametros:
    umbral_oscuro: float = 0.88    # intensidad relativa al fondo local
    radio_max: float = 0.90        # regla 2: nada más allá del 90 % del radio
    area_min: int = 330            # regla 3: tamaño mínimo
    oscuridad_min: float = 0.44    # regla 4: lo más negro son manchas solares
    alargamiento_min: float = 1.8  # regla 5: forma alargada
    union_px: int = 6              # trozos a menos de esta distancia se unen en un filamento


def disco_solar(img):
    """Centro y radio del disco a partir de los píxeles brillantes."""
    m = img > 0.3 * np.percentile(img, 99)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=3))
    lab, _ = ndi.label(m)
    k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    yy, xx = np.nonzero(lab == k)
    return xx.mean(), yy.mean(), np.sqrt(len(xx) / np.pi)


def normalizar(img, cx, cy, R):
    """Divide por el fondo local (mediana en ventanas de ~100 px)."""
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - cx, yy - cy) / R
    dentro = r < 1
    relleno = np.where(dentro, img, np.median(img[dentro]))
    peque = np.asarray(Image.fromarray(relleno.astype(np.float32)).resize((512, 512), Image.BILINEAR))
    fondo = ndi.median_filter(peque, size=25)
    fondo = np.asarray(Image.fromarray(fondo).resize(img.shape[::-1], Image.BILINEAR))
    norm = img / np.maximum(fondo, 1)
    norm[~dentro] = 1
    return norm, r


def observador(norm, r, p):
    """Candidatos: zonas oscuras dentro del disco, uniendo trozos muy cercanos."""
    oscuro = (norm < p.umbral_oscuro) & (r < p.radio_max)
    oscuro = ndi.binary_opening(oscuro, iterations=1)
    # Los trozos cercanos se agrupan con una dilatación, pero la máscara final es la original
    grupos = measure.label(ndi.binary_dilation(oscuro, structure=morphology.disk(p.union_px)))
    return grupos * oscuro


def juez(etiquetas, norm, p):
    """Aplica las reglas a cada candidato y devuelve las máscaras aceptadas."""
    aceptadas = []
    for reg in measure.regionprops(etiquetas):
        if reg.area < p.area_min:
            continue
        y0, x0, y1, x1 = reg.bbox
        vals = norm[y0:y1, x0:x1][reg.image]
        if vals.min() < p.oscuridad_min:
            continue
        if reg.axis_major_length / max(reg.axis_minor_length, 1) < p.alargamiento_min:
            continue
        m = np.zeros(etiquetas.shape, bool)
        m[y0:y1, x0:x1] = reg.image
        aceptadas.append(m)
    return aceptadas


def predecir(ruta_imagen, p=Parametros()):
    img = np.asarray(Image.open(ruta_imagen).convert("L"), float)
    cx, cy, R = disco_solar(img)
    norm, r = normalizar(img, cx, cy, R)
    return juez(observador(norm, r, p), norm, p)
