"""Juez A ("el ojo"): red pequeña que mira el recorte de cada candidato (imagen + silueta).

Aprende con los candidatos de las imágenes de estudio (confianza >= 0,3) y puntúa los del examen.
El maestro es el mismo que el del juez B: qué fracción de los expertos de la imagen dibujó ese
candidato (IoU > 0,5).

Uso (tras tribunal_sim.py y cajas_juez_a.py):
    python3 analisis/juez_a.py <carpeta paso1> <carpeta juez_a>
Guarda <carpeta juez_a>/nota_a_examen.npy (una nota por recorte del examen, mismo orden).
"""
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

CONF_MIN = 0.3


def etiquetas(datos):
    y, w = [], []
    for img in sorted(datos):
        for f in datos[img]["filas"]:
            if f["conf"] >= CONF_MIN:
                y.append(f["aciertos"] / f["expertos"])
                w.append(f["expertos"])
    return np.array(y, np.float32), np.array(w, np.float32)


def entrada(img, sil):
    x = img.astype(np.float32) / 255
    x = (x - x.mean((1, 2), keepdims=True)) / (x.std((1, 2), keepdims=True) + 1e-3)
    return np.stack([x, sil.astype(np.float32) / 255], 1)


def bloque(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))


def red():
    return nn.Sequential(bloque(2, 24), bloque(24, 48), bloque(48, 96), bloque(96, 160), bloque(160, 192),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, 1))


def girar(x):
    """Aumento: una de las 8 simetrías del cuadrado (el sol no tiene arriba ni abajo para un filamento)."""
    k = np.random.randint(4)
    x = torch.rot90(x, k, (2, 3))
    return x.flip(3) if np.random.rand() < 0.5 else x


def entrenar(X, y, w, epocas=14, semilla=0, lote=64):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    m = red()
    opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-3)
    pasos = epocas * int(np.ceil(len(X) / lote))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=pasos)
    X, y, w = map(torch.from_numpy, (X, y, w))
    t = time.time()
    for ep in range(epocas):
        m.train()
        perm = torch.randperm(len(X))
        tot = 0
        for i in range(0, len(X), lote):
            j = perm[i:i + lote]
            p = m(girar(X[j]))[:, 0]
            loss = (nn.functional.binary_cross_entropy_with_logits(p, y[j], reduction="none") * w[j]).sum() / w[j].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(j)
        print(f"  época {ep + 1}: pérdida {tot / len(X):.4f} ({time.time() - t:.0f} s)", flush=True)
    return m


@torch.no_grad()
def predecir(m, X, lote=256):
    m.eval()
    X = torch.from_numpy(X)
    out = []
    for i in range(0, len(X), lote):
        x = X[i:i + lote]
        # Promedio de las 8 simetrías
        p = 0
        for k in range(4):
            for f in (False, True):
                xi = torch.rot90(x, k, (2, 3))
                p = p + torch.sigmoid(m(xi.flip(3) if f else xi)[:, 0])
        out.append((p / 8).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    paso1, carpeta = sys.argv[1], sys.argv[2]
    torch.set_num_threads(4)
    with open(f"{paso1}/tribunal_cache.pkl", "rb") as fh:
        exa, est = pickle.load(fh)
    Xtr = entrada(np.load(f"{carpeta}/recortes_img_estudio.npy"), np.load(f"{carpeta}/siluetas_estudio.npy"))
    Xte = entrada(np.load(f"{carpeta}/recortes_img_examen.npy"), np.load(f"{carpeta}/siluetas_examen.npy"))
    ytr, wtr = etiquetas(est)
    assert len(ytr) == len(Xtr), (len(ytr), len(Xtr))
    notas = []
    for semilla in (0, 1):
        print(f"semilla {semilla}: {len(Xtr)} recortes de estudio", flush=True)
        m = entrenar(Xtr, ytr, wtr, semilla=semilla)
        notas.append(predecir(m, Xte))
        np.save(f"{carpeta}/nota_a_examen.npy", np.mean(notas, 0))
    print("hecho")
