"""¿Qué tienen los filamentos en los que todos los expertos coinciden frente a los que marca uno solo?

Solo imágenes con 2-3 expertos. Para cada filamento de cada experto se mira si los demás
expertos dibujaron el mismo (IoU > 0,5):
  - "todos": lo dibujaron todos los demás.
  - "alguno": solo algunos de los demás (pasa con 3 expertos).
  - "solo": ningún otro (si alguno lo tocó con IoU 0-0,5, es "solo, pero tocado").
Se describen por tamaño, forma, distancia al borde del disco, oscuridad (si la imagen está
en local), cercanía a otro filamento y año.

Uso: python3 analisis/consenso_pequenos.py <carpeta MAGFiLO_1.0_Kaggle_2026> [salida.csv]
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from filamentos import clasico, datos, detector as D  # noqa: E402

TAM = datos.TAM


def rles(anotaciones):
    return [mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], TAM, TAM)) for a in anotaciones]


def disco(ruta):
    img = np.asarray(Image.open(ruta).convert("L"), np.float32)
    cx, cy, R = clasico.disco_solar(img)
    yy, xx = np.mgrid[:TAM, :TAM]
    r = np.hypot(xx - cx, yy - cy) / R
    return img, r, (cx, cy, R)


def main(raiz, salida):
    anot = datos.cargar_anotaciones(raiz)
    carpeta = os.path.join(raiz, "train", "train_images")
    locales = set(os.listdir(carpeta)) if os.path.isdir(carpeta) else set()
    filas = []
    for nombre, por_exp in anot.items():
        if len(por_exp) < 2:
            continue
        exps = sorted(por_exp)
        R = {e: rles(por_exp[e]) for e in exps}
        img = r = None
        if nombre in locales:
            img, r, _ = disco(os.path.join(carpeta, nombre))
            rel = D.brillo_relativo(img, r).astype(np.float32)
        for e in exps:
            propios = R[e]
            if not propios:
                continue
            areas = mask_utils.area(propios)
            cajas = mask_utils.toBbox(propios)
            centros = cajas[:, :2] + cajas[:, 2:] / 2
            otros = [o for o in exps if o != e]
            mejor = {o: (np.asarray(mask_utils.iou(propios, R[o], [0] * len(R[o]))).max(1)
                         if R[o] else np.zeros(len(propios))) for o in otros}
            for i, a in enumerate(por_exp[e]):
                ious = [mejor[o][i] for o in otros]
                n_ok = sum(v > 0.5 for v in ious)
                estado = "todos" if n_ok == len(otros) else ("alguno" if n_ok else "solo")
                d = np.hypot(*(centros - centros[i]).T)
                d[i] = np.inf
                j = int(np.argmin(d)) if len(d) > 1 else None
                m = mask_utils.decode(propios[i]).astype(bool)
                fila = dict(imagen=nombre, experto=e, n_expertos=len(exps), area=float(areas[i]),
                            estado=estado, tocado=max(ious) > 0, iou_max=float(max(ious)),
                            alarg=D.alargamiento(m), n_propios=len(propios),
                            dist_vecino=float(d[j]) if j is not None else np.nan,
                            area_vecino=float(areas[j]) if j is not None else np.nan,
                            anio=int(nombre[:4]), telescopio=nombre[-6])
                if img is not None:
                    fila["radio"] = float(r[m].mean())
                    fila["oscuridad"] = float(np.median(rel[m]))
                    fila["min_oscuro"] = float(np.percentile(rel[m], 5))
                filas.append(fila)
    df = pd.DataFrame(filas)
    df.to_csv(salida, index=False)
    return df


def informe(df):
    pd.set_option("display.width", 160)
    cortes = [0, 330, 1000, 3000, 10000, np.inf]
    nombres = ["<330", "330-1k", "1k-3k", "3k-10k", ">10k"]
    df["tamaño"] = pd.cut(df.area, cortes, labels=nombres, right=False)
    print(f"{len(df)} filamentos de {df.imagen.nunique()} imágenes con 2-3 expertos\n")
    t = pd.crosstab(df["tamaño"], df.estado, normalize="index").mul(100).round(0)
    t["n"] = df.groupby("tamaño", observed=True).size()
    print("% por tamaño (todos = los demás expertos también lo dibujaron)\n", t, "\n")
    peq = df[df.area < 3000].copy()
    print("Filamentos pequeños (<3000 px): mediana de cada rasgo según el consenso")
    cols = ["area", "alarg", "dist_vecino", "area_vecino", "n_propios"] + \
        [c for c in ("radio", "oscuridad", "min_oscuro") if c in peq]
    print(peq.groupby("estado")[cols].median().round(2), "\n")
    print("n con oscuridad medida:", peq.groupby("estado")["oscuridad"].count().to_dict() if "oscuridad" in peq else 0)
    print("\nSolo de un experto: ¿lo tocó otro experto aunque sea mal dibujado?")
    print(peq[peq.estado == "solo"].tocado.value_counts(normalize=True).round(2).to_dict())
    peq["junto_a_grande"] = (peq.dist_vecino < 150) & (peq.area_vecino >= 3000)
    print("\n% 'todos' según si está a menos de 150 px de un filamento grande del mismo experto")
    print(peq.groupby("junto_a_grande").estado.value_counts(normalize=True).unstack().mul(100).round(0))
    peq["alarg_c"] = pd.cut(peq.alarg, [0, 2, 3, 5, np.inf], labels=["<2", "2-3", "3-5", ">5"])
    print("\n% por alargamiento (pequeños)")
    print(pd.crosstab(peq.alarg_c, peq.estado, normalize="index").mul(100).round(0))
    if "oscuridad" in peq:
        o = peq.dropna(subset=["oscuridad"]).copy()
        o["osc_c"] = pd.cut(o.oscuridad, [0, 0.85, 0.9, 0.95, np.inf], labels=["<0,85", "0,85-0,9", "0,9-0,95", ">0,95"])
        print("\n% por oscuridad mediana (pequeños, imágenes en local)")
        t = pd.crosstab(o.osc_c, o.estado, normalize="index").mul(100).round(0)
        t["n"] = o.groupby("osc_c", observed=True).size()
        print(t)
        o["rad_c"] = pd.cut(o.radio, [0, 0.5, 0.8, 0.9, 2], labels=["<0,5", "0,5-0,8", "0,8-0,9", ">0,9"])
        print("\n% por distancia al centro (pequeños)")
        t = pd.crosstab(o.rad_c, o.estado, normalize="index").mul(100).round(0)
        t["n"] = o.groupby("rad_c", observed=True).size()
        print(t)
    print("\n% 'todos' en pequeños por año (ciclo solar)")
    print(peq.groupby("anio").estado.apply(lambda s: round(100 * (s == "todos").mean())).to_dict())
    print("\n% 'todos' en pequeños por experto (quién marca pequeños que nadie más ve)")
    g = peq.groupby("experto").estado.agg(n="size", todos=lambda s: round(100 * (s == "todos").mean()),
                                         solo=lambda s: round(100 * (s == "solo").mean()))
    print(g.sort_values("solo").to_string())


if __name__ == "__main__":
    raiz = sys.argv[1]
    salida = sys.argv[2] if len(sys.argv) > 2 else "consenso_pequenos.csv"
    informe(main(raiz, salida))
