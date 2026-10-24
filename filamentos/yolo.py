"""Ronda 5: detector YOLO11-seg (Ultralytics) en lugar de Mask R-CNN.

Se entrena con los mismos datos que el detector de la ronda 3 (cada experto como ejemplo aparte, imagen a
1024 px) y produce candidatos con el mismo formato que el paso 1 del tribunal, para reutilizar el
refinador, el simulador y la votación sin GPU.
"""
import os

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.sparse.csgraph import connected_components

from .candidatos import de_texto
from .datos import TAM


def tres_vistas(img):
    """Ronda 7 (idea de jhonatan, 24 oct 2026): la misma foto vista de tres formas, una por canal.

    img uint8 [H,W] (gris) -> uint8 [H,W,3]:
      0) la original;
      1) contraste local (CLAHE): realza lo débil sin quemar el centro brillante del disco;
      2) "solo lo oscuro y fino" (black-hat): cuánto más oscuro es cada píxel que su entorno cercano,
         lo que quita el brillo general del disco y deja las rayas oscuras como los filamentos (negras sobre blanco).
    Los tamaños están pensados para 1024 px (filamento típico ~8 px de grueso) y se escalan con la imagen.
    """
    esc = img.shape[0] / 1024
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(img)
    k = int(round(25 * esc)) | 1
    cierre = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    oscuro = cv2.subtract(cierre, img)
    oscuro = 255 - np.clip(oscuro.astype(np.float32) * 6, 0, 255).astype(np.uint8)
    return np.stack([img, clahe, oscuro], -1)


def preparar_datos(datos, anot, tr, val, salida, res=1024, vistas=False):
    """Carpeta en formato YOLO: una imagen a res px por imagen original y un fichero de etiquetas por
    experto (la misma imagen se repite con otro nombre, enlazada, una vez por experto). Las listas de
    entrenamiento y validación solo incluyen las copias por experto, no la imagen original."""
    for parte, nombres in (("train", tr), ("val", val)):
        os.makedirs(f"{salida}/images/{parte}", exist_ok=True)
        os.makedirs(f"{salida}/labels/{parte}", exist_ok=True)
        for n in nombres:
            base = os.path.splitext(n)[0]
            origen = f"{salida}/images/{parte}/{base}.png"
            if not os.path.exists(origen):
                img = Image.open(f"{datos}/train/train_images/{n}").convert("L").resize((res, res), Image.BILINEAR)
                if vistas:
                    img = Image.fromarray(tres_vistas(np.asarray(img)))
                img.save(origen)
            for e, anns in anot[n].items():
                nombre = f"{base}_{e}"
                destino = f"{salida}/images/{parte}/{nombre}.png"
                if not os.path.exists(destino):
                    os.symlink(f"{base}.png", destino)
                with open(f"{salida}/labels/{parte}/{nombre}.txt", "w") as fh:
                    for a in anns:
                        p = np.asarray(a["segmentation"][0], float) / TAM
                        if len(p) >= 6:
                            fh.write("0 " + " ".join(f"{v:.5f}" for v in np.clip(p, 0, 1)) + "\n")
    with open(f"{salida}/datos.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(salida)}\ntrain: lista_train.txt\nval: lista_val.txt\nnames:\n  0: filamento\n")
    for parte in ("train", "val"):
        imgs = sorted(f for f in os.listdir(f"{salida}/images/{parte}") if os.path.islink(f"{salida}/images/{parte}/{f}"))
        with open(f"{salida}/lista_{parte}.txt", "w") as fh:
            fh.write("\n".join(f"./images/{parte}/{f}" for f in imgs) + "\n")
    return f"{salida}/datos.yaml"


def detector(modelo, res=1024, conf_min=0.1, vistas=False):
    """Función img uint8 [res,res] -> (máscaras [N,res,res] bool, confianzas [N]), como detector.detectar.

    vistas=True: el modelo se entrenó con tres_vistas(). Ultralytics lee los PNG de entrenamiento con OpenCV (BGR) y
    trata igual un array que le pasamos, así que se invierte el orden de canales para que coincida."""
    def detectar(img):
        x = tres_vistas(img)[..., ::-1] if vistas else np.stack([img] * 3, -1)
        r = modelo.predict(np.ascontiguousarray(x), imgsz=res, conf=conf_min, retina_masks=True,
                           max_det=300, verbose=False)[0]
        if r.masks is None or not len(r.boxes):
            return np.zeros((0, res, res), bool), np.zeros(0, np.float32)
        m = r.masks.data.cpu().numpy() > 0.5
        c = r.boxes.conf.cpu().numpy().astype(np.float32)
        orden = np.argsort(-c)
        m, c = m[orden], c[orden]
        activas = m.any(axis=(1, 2))
        return m[activas], c[activas]
    return detectar


VISTAS = ((False, False), (True, False), (False, True), (True, True))  # (voltear izquierda-derecha, arriba-abajo)


def detector_tta(modelo, res=1024, conf_min=0.1, vistas=VISTAS):
    """Como detector(), pero mirando la imagen volteada de varias formas (TTA) y juntando las respuestas.

    Cada vista se predice, sus máscaras se devuelven a la posición original y las de distintas vistas que se
    solapan (IoU > 0,5) se agrupan en un candidato. Su confianza es la media de las vistas (0 en las que no
    lo vieron) y su máscara, los píxeles que marcan al menos la mitad de las vistas que lo vieron. Devuelve
    también, por candidato, la confianza de cada vista ('vistas') para probar otras reglas sin GPU."""
    base = detector(modelo, res, conf_min)

    def detectar(img):
        ms, cs, vs = [], [], []
        for v, (lr, ud) in enumerate(vistas):
            x = img[:, ::-1] if lr else img
            x = x[::-1] if ud else x
            m, c = base(np.ascontiguousarray(x))
            m = m[:, :, ::-1] if lr else m
            m = m[:, ::-1] if ud else m
            ms += list(m)
            cs += list(c)
            vs += [v] * len(c)
        if not ms:
            return np.zeros((0, res, res), bool), np.zeros(0, np.float32), []
        rles = [mask_utils.encode(np.asfortranarray(x.astype(np.uint8))) for x in ms]
        vs = np.array(vs)
        enlace = (np.asarray(mask_utils.iou(rles, rles, [0] * len(rles))) > 0.5) & (vs[:, None] != vs[None, :])
        _, et = connected_components(enlace, directed=False)
        masks, confs, extra = [], [], []
        for k in range(et.max() + 1):
            idx = np.flatnonzero(et == k)
            por_vista = np.zeros(len(vistas), np.float32)
            for i in idx:
                por_vista[vs[i]] = max(por_vista[vs[i]], cs[i])
            votos = np.sum([ms[i] for i in idx], axis=0)
            masks.append(votos * 2 >= len(idx))
            confs.append(float(por_vista.mean()))
            extra.append(dict(vistas=[round(float(x), 4) for x in por_vista]))
        orden = np.argsort(-np.array(confs))
        return (np.stack([masks[i] for i in orden]), np.array([confs[i] for i in orden], np.float32),
                [extra[i] for i in orden])
    return detectar


def _sin_repetidos(cands, clave):
    cands = sorted(cands, key=lambda c: -c["conf"])
    rles = [de_texto(c[clave]) for c in cands]
    if not rles:
        return [], []
    dup = np.triu(np.asarray(mask_utils.iou(rles, rles, [0] * len(rles))) > 0.5, 1).any(0)
    return [c for c, d in zip(cands, dup) if not d], [r for r, d in zip(rles, dup) if not d]


def seleccion(cands, umbral, clave="rle_ref", radio_max=0.95):
    """Juez de la ronda 5 para una imagen: confianza >= umbral, fuera del borde, sin repetidos, unir los que se tocan.
    Devuelve la lista de RLE finales."""
    cs, rles = _sin_repetidos(cands, clave)
    sel = [r for c, r in zip(cs, rles) if c["conf"] >= umbral and c["radio"] <= radio_max]
    if len(sel) > 1:
        toca = np.asarray(mask_utils.iou(sel, sel, [0] * len(sel))) > 0
        _, et = connected_components(toca, directed=False)
        sel = [mask_utils.merge([sel[j] for j in np.flatnonzero(et == k)]) for k in range(et.max() + 1)]
    return sel


def nota_umbral(cands_por_imagen, reales_por_imagen, umbral, clave="rle_ref", radio_max=0.95, por_tamano=False):
    """PQ con el juez de la ronda 3 aplicado a los candidatos: confianza >= umbral, unir los que se tocan.

    Devuelve dict con pq, sq, tp, fp, fn y, si por_tamano, los escapados por tamaño del filamento real."""
    tp = fp = fn = 0
    suma = 0.0
    tamanos = (0, 1000, 3000, 10 ** 9)
    escapados = np.zeros(3, int)
    for n, cands in cands_por_imagen.items():
        sel = seleccion(cands, umbral, clave, radio_max)
        for reales in reales_por_imagen[n]:
            if not sel or not reales:
                fp += len(sel)
                fn += len(reales)
                acertados = np.zeros(len(reales), bool)
            else:
                m = np.asarray(mask_utils.iou(sel, reales, [0] * len(reales)))
                hit = m > 0.5
                acertados = hit.any(0)
                tp += int(acertados.sum())
                suma += float(m[hit].sum())
                fn += int((~acertados).sum())
                fp += len(sel) - int(hit.any(1).sum())
            if por_tamano and len(reales):
                areas = mask_utils.area(reales)
                escapados += np.histogram(areas[~acertados], tamanos)[0]
    den = tp + 0.5 * fp + 0.5 * fn
    out = dict(pq=suma / den if den else 0.0, sq=suma / tp if tp else 0.0, tp=tp, fp=fp, fn=fn)
    if por_tamano:
        out["escapados_menos_1k"], out["escapados_1k_3k"], out["escapados_mas_3k"] = map(int, escapados)
    return out


def reales_rle(anot_imagen):
    """{experto: [anotación]} -> [[RLE de cada filamento] por experto], sin pasar por máscaras."""
    return [[mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], TAM, TAM)) for a in anns]
            for anns in anot_imagen.values()]
