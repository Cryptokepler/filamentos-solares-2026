"""Detector + refinador: cada filamento como un objeto separado.

1. Detector (Mask R-CNN, preentrenado en COCO): a 1024 px propone cada filamento con
   una caja, una confianza y una silueta. Partir y unir lo decide él, no una regla.
2. Refinador (U-Net pequeña): recorta cada filamento detectado en la imagen original
   de 2048 px, con margen alrededor, y redibuja su borde a 256x256.

Arquitectura inspirada en el notebook público de Nomannic19 ("Mask R-CNN + U-Net
Refiner"); el código es nuestro.
"""
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components

TAM = 2048
RECORTE = 256


# ---------------------------------------------------------------- detector

def construir_detector(preentrenado=True, res=1024):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign

    pesos = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if preentrenado else None
    m = maskrcnn_resnet50_fpn_v2(weights=pesos, weights_backbone=None)
    # Dos clases: fondo y filamento
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(
        m.roi_heads.mask_predictor.conv5_mask.in_channels, m.roi_heads.mask_predictor.conv5_mask.out_channels, 2)
    # Cajas desde 16 px y alargadas (los filamentos miden 2-6 veces más de largo que de ancho)
    m.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                             aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    # Siluetas a 28x28 en vez de 14x14: más detalle en filamentos finos
    m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    m.roi_heads.detections_per_img = 100
    m.transform.min_size, m.transform.max_size = (res,), res
    return m


def cargar_gris(ruta, res):
    return np.asarray(Image.open(ruta).convert("L").resize((res, res), Image.BILINEAR), np.uint8)


def mascaras_res(anotaciones, res, tam=TAM):
    """Una máscara booleana [res,res] por filamento de un anotador.

    Se dibuja a 2048 con pycocotools (como la métrica oficial) y se reduce promediando.
    Dibujar el polígono directamente a baja resolución con PIL engorda los bordes: con
    máscaras perfectas, la PQ caía de 0,89 a 0,82 en una prueba con 15 imágenes."""
    out = []
    for a in anotaciones:
        rles = mask_utils.frPyObjects(a["segmentation"], tam, tam)
        m = mask_utils.decode(mask_utils.merge(rles)).astype(np.float32)
        m = np.asarray(Image.fromarray(m).resize((res, res), Image.BILINEAR)) > 0.5
        if m.any():
            out.append(m)
    return out


def _fotometria(img):
    """Cambios de brillo, contraste y gamma solo dentro del disco (fuera es negro)."""
    dentro = img > 0.01
    if np.random.rand() < 0.7:
        img[dentro] = img[dentro] ** np.random.uniform(0.8, 1.25)
    if np.random.rand() < 0.7:
        media = img[dentro].mean()
        img[dentro] = ((img[dentro] - media) * np.random.uniform(0.85, 1.15) + media) * np.random.uniform(0.9, 1.1)
    return np.clip(img, 0, 1)


class DatosDetector(torch.utils.data.Dataset):
    """Un ejemplo por (imagen, anotador): el detector aprende con siluetas nítidas de un experto,
    no con la media borrosa. Las versiones de una misma imagen comparten la imagen en memoria y
    las máscaras se dibujan al vuelo desde los polígonos (guardarlas todas no cabe en RAM)."""

    def __init__(self, ejemplos, res, aumentar):
        self.ejemplos, self.res, self.aumentar = ejemplos, res, aumentar  # [(imagen uint8 [res,res], anotaciones)]

    def __len__(self):
        return len(self.ejemplos)

    def __getitem__(self, i):
        img, anotaciones = self.ejemplos[i]
        return objetivo_aumentado(img, mascaras_res(anotaciones, self.res), self.aumentar)


def objetivo_aumentado(img, mascaras, aumentar):
    """img uint8 [H,W] + máscaras booleanas -> (tensor imagen, objetivo de Mask R-CNN), con giros y
    cambios de brillo si aumentar. Lo comparten el detector general y el especialista."""
    img = img.astype(np.float32) / 255
    m = np.stack(mascaras) if mascaras else np.zeros((0,) + img.shape, bool)
    if aumentar:
        k = np.random.randint(4)
        img, m = np.rot90(img, k), np.rot90(m, k, (1, 2))
        if np.random.rand() < 0.5:
            img, m = img[:, ::-1], m[:, :, ::-1]
        img = _fotometria(img.copy())
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    cajas = np.zeros((len(m), 4), np.float32)
    for j, mj in enumerate(m):
        ys, xs = np.nonzero(mj)
        cajas[j] = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    objetivo = {"boxes": torch.from_numpy(cajas), "labels": torch.ones(len(m), dtype=torch.int64),
                "masks": torch.from_numpy(m.astype(np.uint8))}
    return torch.from_numpy(img)[None].repeat(3, 1, 1), objetivo


def juntar(lote):
    return tuple(zip(*lote))


@torch.no_grad()
def detectar(modelo, img, disp, umbral_mascara=0.5):
    """img uint8 [res,res] -> (máscaras booleanas [N,res,res], confianzas [N]) sin filtrar por confianza."""
    modelo.eval()
    x = torch.from_numpy(img.astype(np.float32) / 255)[None].repeat(3, 1, 1).to(disp)
    s = modelo([x])[0]
    m = (s["masks"][:, 0] >= umbral_mascara).cpu().numpy()
    conf = s["scores"].float().cpu().numpy()
    activas = m.any(axis=(1, 2))
    return m[activas], conf[activas]


def a_rle(m):
    return mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))


def grupos_solapados(mascaras):
    """Índices agrupados: dos detecciones que se tocan (IoU > 0) van al mismo grupo."""
    n = len(mascaras)
    if n < 2:
        return [[i] for i in range(n)]
    rles = [a_rle(m) for m in mascaras]
    toca = np.asarray(mask_utils.iou(rles, rles, [0] * n)) > 0
    _, etiqueta = connected_components(toca, directed=False)
    return [list(np.flatnonzero(etiqueta == k)) for k in range(etiqueta.max() + 1)]


def a_2048(m):
    """Amplía con suavizado bilineal: el borde queda liso en vez de en escalones de 2 px."""
    return np.asarray(Image.fromarray(m.astype(np.float32)).resize((TAM, TAM), Image.BILINEAR)) > 0.5


def brillo_relativo(img, r):
    """Brillo de cada píxel dividido por el fondo local (mediana en ventanas de ~100 px a 2048).

    Mismo cálculo que el análisis de manchas (reglas 4 y 5): corrige el oscurecimiento hacia el
    borde y la iluminación desigual. 1 = como el fondo; 0,5 = la mitad de brillante.
    """
    img = img.astype(np.float32)
    dentro = r < 1
    relleno = np.where(dentro, img, np.median(img[dentro]))
    peque = np.asarray(Image.fromarray(relleno).resize((512, 512), Image.BILINEAR))
    fondo = ndi.median_filter(peque, size=25)
    fondo = np.asarray(Image.fromarray(fondo).resize(img.shape[::-1], Image.BILINEAR))
    rel = img / np.maximum(fondo, 1)
    rel[~dentro] = 1
    return rel.astype(np.float16)


def alargamiento(m):
    """Eje mayor / eje menor de la elipse equivalente (como skimage regionprops)."""
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return 1.0
    l = np.linalg.eigvalsh(np.cov(np.stack([xs, ys]).astype(np.float64)))
    return float(np.sqrt(max(l[1], 1e-9) / max(l[0], 1e-9)))


def instancias(mascaras, conf, r, umbral=0.6, fusionar=True, area_min=0, radio_max=0.95,
               rel=None, oscuro_min=None, alarg_max=None):
    """Detecciones a [res,res] -> lista de máscaras a 2048 tras el juez.

    fusionar: las detecciones que se solapan se unen en un solo filamento (Mask R-CNN
    a veces propone el mismo filamento dos veces, o dos trozos de él).
    r: distancia al centro en radios a resolución res.
    oscuro_min (regla 4): se descarta si su punto más oscuro baja de este brillo relativo
        (rel = brillo_relativo); lo negrísimo suele ser una mancha solar.
    alarg_max (regla 5): se descarta si es más alargado que esto (hilos larguísimos).
    """
    keep = conf >= umbral
    mascaras = mascaras[keep]
    if fusionar:
        mascaras = [np.logical_or.reduce(mascaras[g]) for g in grupos_solapados(mascaras)]
    else:
        # Sin fusionar, un píxel solo puede pertenecer a un filamento: gana el de más confianza
        ocupado = np.zeros(r.shape, bool)
        limpias = []
        for m in mascaras:
            m = m & ~ocupado
            ocupado |= m
            if m.any():
                limpias.append(m)
        mascaras = limpias
    out = []
    escala2 = (TAM / r.shape[0]) ** 2
    for m in mascaras:
        if m.sum() * escala2 < area_min or r[m].mean() > radio_max:
            continue
        if oscuro_min is not None and float(rel[m].min()) < oscuro_min:
            continue
        if alarg_max is not None and alargamiento(m) > alarg_max:
            continue
        out.append(a_2048(m))
    return out


# ---------------------------------------------------------------- refinador

def caja_cuadrada(m, margen=1.8, minimo=96, tam=TAM):
    """Cuadrado centrado en el filamento, margen veces su lado mayor (contexto alrededor)."""
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    lado = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * margen))
    lado = min(max(lado, minimo), tam)
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x0, y0 = min(max(x0, 0), tam - lado), min(max(y0, 0), tam - lado)
    return x0, y0, x0 + lado, y0 + lado


def recortar(img, caja, interp=Image.BILINEAR):
    x0, y0, x1, y1 = caja
    return np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((RECORTE, RECORTE), interp))


def recortes_entreno(img2048, mascaras2048, aumentar):
    """Pares (recorte de imagen, recorte de máscara) a 256x256, uno por filamento.

    Con aumentar=True el margen y el centro varían un poco, para que el refinador no dependa
    de que el detector encuadre el filamento perfectamente."""
    out = []
    for m in mascaras2048:
        margen = np.random.uniform(1.5, 2.2) if aumentar else 1.8
        caja = caja_cuadrada(m, margen)
        if aumentar:
            lado = caja[2] - caja[0]
            dx, dy = np.random.randint(-lado // 10, lado // 10 + 1, 2)
            x0, y0 = min(max(caja[0] + dx, 0), TAM - lado), min(max(caja[1] + dy, 0), TAM - lado)
            caja = (x0, y0, x0 + lado, y0 + lado)
        out.append((recortar(img2048, caja), recortar(m.astype(np.uint8), caja, Image.NEAREST)))
    return out


class DatosRecortes(torch.utils.data.Dataset):
    def __init__(self, imagenes, mascaras):
        self.x, self.y = imagenes, mascaras  # uint8 [N,256,256]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, y = self.x[i].astype(np.float32) / 255, self.y[i].astype(np.float32)
        k = np.random.randint(4)
        x, y = np.rot90(x, k), np.rot90(y, k)
        if np.random.rand() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = np.clip(x * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.04, 0.04), 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(x))[None].repeat(3, 1, 1),
                torch.from_numpy(np.ascontiguousarray(y))[None])


@torch.no_grad()
def refinar(refinador, img2048, mascaras2048, disp, umbral=0.85, lote=32):
    """Redibuja cada filamento. Si el refinador no ve nada, se queda la silueta del detector."""
    if not mascaras2048:
        return []
    refinador.eval()
    cajas = [caja_cuadrada(m) for m in mascaras2048]
    recs = np.stack([recortar(img2048, c) for c in cajas]).astype(np.float32) / 255
    probs = []
    for i in range(0, len(recs), lote):
        x = torch.from_numpy(recs[i:i + lote])[:, None].repeat(1, 3, 1, 1).to(disp)
        # Promedio de 4 vistas volteadas (TTA)
        p = 0
        for dims in ((), (3,), (2,), (2, 3)):
            xi = x.flip(dims) if dims else x
            pi = torch.sigmoid(refinador(xi).float())
            p = p + (pi.flip(dims) if dims else pi)
        probs.append((p / 4)[:, 0].cpu().numpy())
    probs = np.concatenate(probs)
    out = []
    for m, (x0, y0, x1, y1), p in zip(mascaras2048, cajas, probs):
        nueva = np.asarray(Image.fromarray(p).resize((x1 - x0, y1 - y0), Image.BILINEAR)) >= umbral
        if not nueva.any():
            out.append(m)
            continue
        # Nos quedamos con la parte conectada que toca la silueta del detector: el recorte puede
        # incluir trozos de filamentos vecinos que no son este
        etiquetas, _ = ndi.label(nueva)
        ids = np.unique(etiquetas[m[y0:y1, x0:x1] & (etiquetas > 0)])
        if not len(ids):
            out.append(m)
            continue
        nueva = np.isin(etiquetas, ids)
        completa = np.zeros((TAM, TAM), bool)
        completa[y0:y1, x0:x1] = nueva
        out.append(completa)
    # Si dos filamentos refinados se pisan, se unen (igual que en el detector)
    return [np.logical_or.reduce([out[i] for i in g]) for g in grupos_solapados(out)]
