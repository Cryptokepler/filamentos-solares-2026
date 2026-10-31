"""Refinador más exigente = silueta más fina (31 oct 2026, CPU).

Uso:
    python3 analisis/umbral_refinador.py <carpeta MAGFiLO> <candidatos_examen.json> <refinador.pt> <salida_dir> [conf_min]

Vuelve a pasar el refinador (U-Net de la ronda 3, con sus 4 vistas volteadas) sobre la silueta del detector de cada
candidato y guarda un candidatos_examen_u{umbral}.json por umbral, con 'rle_ref' redibujado. Con 0,5 debe salir lo
mismo que el json de entrada.
"""
import json
import os
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from filamentos import detector as D  # noqa: E402
from filamentos.candidatos import a_texto  # noqa: E402

UMBRALES = (0.4, 0.5, 0.6, 0.7, 0.8)


def probabilidades(ref, img, mascaras, lote=16):
    cajas = [D.caja_cuadrada(m) for m in mascaras]
    recs = np.stack([D.recortar(img, c) for c in cajas]).astype(np.float32) / 255
    out = []
    with torch.no_grad():
        for i in range(0, len(recs), lote):
            x = torch.from_numpy(recs[i:i + lote])[:, None].repeat(1, 3, 1, 1)
            p = 0
            for dims in ((), (3,), (2,), (2, 3)):
                xi = x.flip(dims) if dims else x
                pi = torch.sigmoid(ref(xi).float())
                p = p + (pi.flip(dims) if dims else pi)
            out.append((p / 4)[:, 0].numpy())
    return cajas, np.concatenate(out)


def redibujar(m, caja, p, umbral):
    """Igual que detector.refinar: umbral, parte conectada que toca la silueta del detector."""
    x0, y0, x1, y1 = caja
    nueva = np.asarray(Image.fromarray(p).resize((x1 - x0, y1 - y0), Image.BILINEAR)) >= umbral
    if not nueva.any():
        return m
    et, _ = ndi.label(nueva)
    ids = np.unique(et[m[y0:y1, x0:x1] & (et > 0)])
    if not len(ids):
        return m
    completa = np.zeros((D.TAM, D.TAM), bool)
    completa[y0:y1, x0:x1] = np.isin(et, ids)
    return completa


if __name__ == "__main__":
    raiz, ruta, pesos, salida = sys.argv[1:5]
    conf_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    torch.set_num_threads(os.cpu_count())
    ref = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1)
    ref.load_state_dict(torch.load(pesos, map_location="cpu"))
    ref.eval()
    cand = json.load(open(ruta))
    res = {u: {} for u in UMBRALES}
    t = time.time()
    for k, n in enumerate(sorted(cand)):
        cs = [c for c in cand[n] if c["conf"] >= conf_min]
        for u in UMBRALES:
            res[u][n] = []
        if not cs:
            continue
        img = np.asarray(Image.open(f"{raiz}/train/train_images/{n}").convert("L"), np.uint8)
        ms = [D.mask_utils.decode({"size": [D.TAM, D.TAM], "counts": c["rle"].encode()}).astype(bool) for c in cs]
        cajas, probs = probabilidades(ref, img, ms)
        for u in UMBRALES:
            for c, m, caja, p in zip(cs, ms, cajas, probs):
                r = redibujar(m, caja, p, u)
                res[u][n].append({**c, "rle_ref": a_texto(r), "area_ref": int(r.sum())})
        if k % 10 == 0:
            print(f"{k + 1}/{len(cand)} imágenes, {time.time() - t:.0f} s", flush=True)
    os.makedirs(salida, exist_ok=True)
    for u in UMBRALES:
        json.dump(res[u], open(f"{salida}/candidatos_examen_u{u}.json", "w"), separators=(",", ":"))
