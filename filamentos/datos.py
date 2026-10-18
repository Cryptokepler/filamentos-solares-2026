"""Lectura de MAGFiLO y conversión entre polígonos, máscaras y RLE."""
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_utils

TAM = 2048


def cargar_anotaciones(raiz):
    """Devuelve {nombre_imagen: {anotador: [anotación COCO, ...]}}.

    Una imagen puede tener hasta 3 registros, uno por anotador. El id del registro
    es "GGGGAA-<nombre>"; usamos GGGGAA como identificador del anotador.
    """
    with open(os.path.join(raiz, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")) as f:
        coco = json.load(f)
    registro = {i["id"]: (i["file_name"], i["id"].split("-")[0]) for i in coco["images"]}
    out = defaultdict(lambda: defaultdict(list))
    for reg_id, (nombre, anot) in registro.items():
        out[nombre][anot]  # imágenes anotadas sin filamentos también cuentan
    for a in coco["annotations"]:
        nombre, anot = registro[a["image_id"]]
        out[nombre][anot].append(a)
    return {k: dict(v) for k, v in out.items()}


def poligono_a_mascara(segmentacion, alto=TAM, ancho=TAM):
    rles = mask_utils.frPyObjects(segmentacion, alto, ancho)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def mascaras_de(anotaciones):
    return [poligono_a_mascara(a["segmentation"]) for a in anotaciones]


def mascara_a_rle(mascara):
    """Cadena de 'counts' RLE de COCO, el formato de la columna segmentation_rle."""
    rle = mask_utils.encode(np.asfortranarray(mascara.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def mascara_a_poligonos(mascara, tolerancia=1.0):
    """Contornos de una máscara como listas [x0, y0, x1, y1, ...] (formato COCO), para el visor."""
    from scipy import ndimage as ndi
    from skimage import measure

    ys, xs = ndi.find_objects(mascara.astype(np.uint8))[0]
    local = np.pad(mascara[ys, xs], 1)
    poligonos = []
    for c in measure.find_contours(local.astype(np.uint8), 0.5):
        c = measure.approximate_polygon(c, tolerancia)
        if len(c) >= 3:
            poligonos.append([round(float(v), 1) for y, x in c for v in (x + xs.start - 1, y + ys.start - 1)])
    return poligonos


def rle_a_mascara(counts, alto=TAM, ancho=TAM):
    return mask_utils.decode({"size": [alto, ancho], "counts": counts.encode("ascii")}).astype(bool)
