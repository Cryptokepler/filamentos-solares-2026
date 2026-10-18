"""Observador con red neuronal: U-Net que predice, píxel a píxel, la probabilidad de filamento.

Entrada de 3 canales a resolución reducida:
  0. imagen original normalizada a [0, 1]
  1. imagen dividida por su fondo local (resalta lo oscuro respecto al entorno)
  2. distancia al centro del disco en radios (ayuda con la regla del borde)
Objetivo: media de las máscaras de todos los anotadores (0, 1/3, 1/2, 2/3 o 1).
"""
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from . import clasico


def preparar_imagen(ruta, res):
    """Devuelve (entrada [3,res,res] float32, r [res,res])."""
    img = np.asarray(Image.open(ruta).convert("L").resize((res, res), Image.BILINEAR), np.float32)
    cx, cy, R = clasico.disco_solar(img)
    yy, xx = np.indices(img.shape)
    r = np.hypot(xx - cx, yy - cy) / R
    dentro = r < 1
    relleno = np.where(dentro, img, np.median(img[dentro]))
    fondo = ndi.median_filter(relleno, size=max(3, res // 20))
    norm = np.clip(img / np.maximum(fondo, 1), 0, 1.5)
    norm[~dentro] = 1
    entrada = np.stack([img / 255.0, norm / 1.5, np.clip(r, 0, 1.2) / 1.2]).astype(np.float32)
    return entrada, r.astype(np.float32)


def objetivo_suave(anotadores, res, tam=2048):
    """Media de las máscaras de cada anotador, dibujadas directamente a resolución res."""
    escala = res / tam
    acum = np.zeros((res, res), np.float32)
    for anns in anotadores.values():
        lienzo = Image.new("L", (res, res), 0)
        d = ImageDraw.Draw(lienzo)
        for a in anns:
            for p in a["segmentation"]:
                if len(p) >= 6:
                    d.polygon([(p[i] * escala, p[i + 1] * escala) for i in range(0, len(p), 2)], fill=1)
        acum += np.asarray(lienzo, np.float32)
    return acum / max(len(anotadores), 1)


class DatosEntreno(torch.utils.data.Dataset):
    def __init__(self, entradas, objetivos, aumentar):
        self.x, self.y, self.aumentar = entradas, objetivos, aumentar

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i], self.y[i][None]
        if self.aumentar:
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k, (1, 2))
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            x = x.copy()
            x[0] = np.clip(x[0] * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def perdida(logits, y):
    """BCE + Dice suave: BCE aprende píxel a píxel, Dice compensa que el filamento ocupa poco."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def predecir_prob(modelo, entrada, dispositivo, tta=True):
    """Probabilidad [res,res]; con TTA promedia las 8 rotaciones/volteos."""
    x = torch.from_numpy(entrada)[None].to(dispositivo)
    vistas = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    acum = 0
    for k, f in vistas:
        xi = torch.rot90(x, k, (2, 3))
        if f:
            xi = xi.flip(3)
        p = torch.sigmoid(modelo(xi))
        if f:
            p = p.flip(3)
        acum = acum + torch.rot90(p, -k, (2, 3))
    return (acum / len(vistas))[0, 0].float().cpu().numpy()


def a_instancias(prob, r, umbral=0.5, area_min=330, area_max=60000, radio_max=0.95, union_px=0, tam=2048):
    """Probabilidad a resolución reducida -> lista de máscaras booleanas a 2048x2048."""
    prob_g = np.asarray(Image.fromarray(prob).resize((tam, tam), Image.BILINEAR))
    r_g = np.asarray(Image.fromarray(r).resize((tam, tam), Image.BILINEAR))
    # Solo dentro del disco: fuera no hay filamentos y una red poco entrenada puede dar ruido
    fil = (prob_g > umbral) & (r_g < 1.0)
    if union_px:
        grupos, n = ndi.label(ndi.binary_dilation(fil, iterations=union_px))
        etiquetas = grupos * fil
    else:
        etiquetas, n = ndi.label(fil, structure=np.ones((3, 3)))
    mascaras = []
    for k, sl in enumerate(ndi.find_objects(etiquetas), start=1):
        if sl is None:
            continue
        m_local = etiquetas[sl] == k
        area = m_local.sum()
        # El filamento anotado más grande mide ~38 000 px; algo mucho mayor es un error
        if area < area_min or area > area_max:
            continue
        if r_g[sl][m_local].mean() > radio_max:
            continue
        m = np.zeros((tam, tam), bool)
        m[sl] = m_local
        mascaras.append(m)
    return mascaras
