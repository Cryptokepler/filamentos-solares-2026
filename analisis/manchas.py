"""Compara las manchas oscuras que los expertos dibujaron con las que ignoraron.

Uso: python3 manchas.py <carpeta MAGFiLO_1.0_Kaggle_2026> <salida.csv>
Solo analiza las imágenes de train presentes en la carpeta.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure

UMBRAL_OSCURO = 0.88   # intensidad relativa al fondo local para considerar "oscuro"
AREA_MIN = 60          # px a 2048: por debajo es ruido


def disco_solar(img):
    """Centro y radio del disco a partir de los píxeles brillantes."""
    m = img > 0.3 * np.percentile(img, 99)
    m = ndi.binary_fill_holes(ndi.binary_opening(m, iterations=3))
    lab, n = ndi.label(m)
    k = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    yy, xx = np.nonzero(lab == k)
    return xx.mean(), yy.mean(), np.sqrt(len(xx) / np.pi)


def normalizar(img, cx, cy, R):
    """Divide por el fondo local (mediana en ventanas de ~100 px).

    Corrige a la vez el oscurecimiento hacia el borde y la iluminación desigual (nubes, velos).
    """
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - cx, yy - cy) / R
    dentro = r < 1
    # Fuera del disco se rellena con la mediana del disco para no contaminar el fondo cercano al borde.
    relleno = np.where(dentro, img, np.median(img[dentro]))
    peque = np.asarray(Image.fromarray(relleno.astype(np.float32)).resize((512, 512), Image.BILINEAR))
    fondo = ndi.median_filter(peque, size=25)
    fondo = np.asarray(Image.fromarray(fondo).resize(img.shape[::-1], Image.BILINEAR))
    norm = img / np.maximum(fondo, 1)
    norm[~dentro] = 1
    return norm, r


def mascara_anotada(anns, shape):
    lienzo = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(lienzo)
    for a in anns:
        for poly in a["segmentation"]:
            if len(poly) >= 6:
                d.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.array(lienzo, bool)


def rasgos(region, norm, r):
    coords = region.coords
    vals = norm[coords[:, 0], coords[:, 1]]
    menor = max(region.axis_minor_length, 1)
    return {
        "area": region.area,
        "alargamiento": region.axis_major_length / menor,
        "oscuridad_media": float(vals.mean()),
        "oscuridad_min": float(vals.min()),
        "solidez": region.solidity,
        "dist_centro": float(r[coords[:, 0], coords[:, 1]].mean()),
    }


def main(raiz, salida):
    with open(os.path.join(raiz, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")) as f:
        coco = json.load(f)
    nombre_de = {i["id"]: i["file_name"] for i in coco["images"]}
    anns_por_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_por_img[nombre_de[a["image_id"]]].append(a)

    carpeta = os.path.join(raiz, "train", "train_images")
    filas = []
    for nombre in sorted(os.listdir(carpeta)):
        img = np.asarray(Image.open(os.path.join(carpeta, nombre)).convert("L"), float)
        cx, cy, R = disco_solar(img)
        norm, r = normalizar(img, cx, cy, R)
        anotada = mascara_anotada(anns_por_img[nombre], img.shape)

        oscuro = (norm < UMBRAL_OSCURO) & (r < 0.97)
        oscuro = ndi.binary_opening(oscuro, iterations=1)
        lab = measure.label(oscuro, connectivity=2)
        for reg in measure.regionprops(lab):
            if reg.area < AREA_MIN:
                continue
            c = reg.coords
            solape = anotada[c[:, 0], c[:, 1]].mean()
            fila = {"imagen": nombre, "tipo": "dibujada" if solape > 0.3 else "ignorada",
                    "x": int(reg.centroid[1]), "y": int(reg.centroid[0])}
            fila.update(rasgos(reg, norm, r))
            filas.append(fila)

    with open(salida, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"{len(filas)} manchas oscuras en {len(os.listdir(carpeta))} imágenes -> {salida}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
