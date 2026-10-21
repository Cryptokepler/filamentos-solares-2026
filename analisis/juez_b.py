"""Juez B ("la ficha"): aprende de las imágenes de estudio y se mide en el examen.

Uso (tras analisis/tribunal_sim.py, que deja la caché):
    python3 analisis/juez_b.py <carpeta MAGFiLO_1.0_Kaggle_2026> <carpeta con candidatos_*.json>
"""
import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.tribunal_sim import RASGOS, cargar, iou, nota  # noqa: E402

OBVIO, BASURA = 0.95, 0.3


def expandir(datos):
    """Una fila por candidato y resultado (acierto / fallo), con peso = nº de expertos de cada caso."""
    X, y, w = [], [], []
    for d in datos.values():
        for f in d["filas"]:
            x = [f[r] for r in RASGOS]
            if f["aciertos"]:
                X.append(x); y.append(1); w.append(f["aciertos"])
            if f["expertos"] - f["aciertos"]:
                X.append(x); y.append(0); w.append(f["expertos"] - f["aciertos"])
    return np.array(X, float), np.array(y), np.array(w, float)


def puntuar(modelo, datos, rasgos):
    cols = [RASGOS.index(r) for r in rasgos]
    for d in datos.values():
        if d["filas"]:
            X = np.array([[f[r] for r in RASGOS] for f in d["filas"]], float)[:, cols]
            for f, p in zip(d["filas"], modelo.predict_proba(X)[:, 1]):
                f["nota_b"] = float(p)


def entrenar(datos, rasgos, semilla=0):
    X, y, w = expandir(datos)
    cols = [RASGOS.index(r) for r in rasgos]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                       min_samples_leaf=40, l2_regularization=1.0, random_state=semilla)
    m.fit(X[:, cols], y, sample_weight=w)
    return m


def receta(umbral, no_tocar=True, clave="nota_b"):
    """Obvios pasan, basura fuera, dudosos al tribunal. no_tocar: un dudoso aprobado que toca a uno
    ya aceptado de más confianza se descarta (para no estropear al bueno al unirlos)."""
    def elegir(filas, rles=None):
        return [i for i, f in enumerate(filas)
                if f["conf"] >= OBVIO or (f["conf"] >= BASURA and f[clave] >= umbral)]
    return elegir


def aplicar(datos, elegir, no_tocar):
    """Envuelve elegir para aplicar no_tocar con las siluetas de cada imagen."""
    if not no_tocar:
        return elegir
    rles_de = {id(d["filas"]): d["rles"] for d in datos.values()}

    def envuelto(filas):
        idx = elegir(filas)
        rles = rles_de[id(filas)]
        aceptados = []
        for i in sorted(idx, key=lambda i: -filas[i]["conf"]):
            if filas[i]["conf"] >= OBVIO or filas[i]["conf"] >= 0.8 and filas[i].get("nota_b", 1) >= 0:
                if filas[i]["conf"] < OBVIO and aceptados and \
                        (iou([rles[i]], [rles[j] for j in aceptados]) > 0).any() and filas[i]["conf"] < 0.8:
                    continue
                aceptados.append(i)
            elif not aceptados or not (iou([rles[i]], [rles[j] for j in aceptados]) > 0).any():
                aceptados.append(i)
        return aceptados
    return envuelto


def calibracion(datos, clave="nota_b"):
    p, y, w = [], [], []
    for d in datos.values():
        for f in d["filas"]:
            if BASURA <= f["conf"] < OBVIO:
                p.append(f[clave]); y.append(f["aciertos"]); w.append(f["expertos"])
    p, y, w = map(np.array, (p, y, w))
    print("  calibración (dudosos del examen): nota -> % real de acierto")
    for a, b in ((0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .8), (.8, 1.01)):
        s = (p >= a) & (p < b)
        if s.any():
            print(f"    {a:.1f}-{min(b, 1):.1f}: {s.sum():4d} candidatos, nota media {p[s].mean():.2f}, "
                  f"acierto real {y[s].sum() / w[s].sum():.2f}")
    yy = np.repeat(np.r_[np.ones(len(y)), np.zeros(len(y))], np.r_[y, w - y].astype(int))
    pp = np.repeat(np.r_[p, p], np.r_[y, w - y].astype(int))
    return roc_auc_score(yy, pp)


if __name__ == "__main__":
    raiz, carpeta = sys.argv[1], sys.argv[2]
    with open(f"{carpeta}/tribunal_cache.pkl", "rb") as fh:
        exa, est = pickle.load(fh)
    base = nota(exa, lambda fs: [i for i, f in enumerate(fs) if f["conf"] >= 0.8])
    print(f"Base (confianza >= 0,8, como la ronda 3): PQ {base[0]:.4f}  TP {base[2]} FP {base[3]} FN {base[4]}")

    imgs = sorted(exa)
    rng = np.random.default_rng(0)
    rng.shuffle(imgs)
    mitad_a, mitad_b = imgs[:60], imgs[60:]

    for nombre, rasgos in (("solo confianza", ["conf"]),
                           ("ficha completa", RASGOS),
                           ("ficha sin confianza", [r for r in RASGOS if r != "conf"])):
        m = entrenar(est, rasgos)
        puntuar(m, exa, rasgos)
        print(f"\n=== Juez B: {nombre} (aprende en estudio, se mide en examen) ===")
        auc = calibracion(exa)
        print(f"  AUC en los dudosos: {auc:.3f}")
        for no_tocar in (False, True):
            filas = []
            for t in (0.2, 0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5, 0.6):
                el = aplicar(exa, receta(t), no_tocar)
                filas.append((t, nota(exa, el), nota(exa, el, imagenes=mitad_a), nota(exa, el, imagenes=mitad_b)))
            print(f"  no_tocar={no_tocar}")
            for t, r, ra, rb in filas:
                print(f"    umbral {t:.2f}: PQ {r[0]:.4f} TP {r[2]} FP {r[3]} FN {r[4]} | mitad A {ra[0]:.4f} B {rb[0]:.4f}")
            # Honesto: umbral elegido en una mitad, medido en la otra
            ta = max(filas, key=lambda x: x[2][0])[0]
            tb = max(filas, key=lambda x: x[3][0])[0]
            ba = nota(exa, aplicar(exa, receta(0), False) if False else (lambda fs: [i for i, f in enumerate(fs) if f["conf"] >= 0.8]), imagenes=mitad_b)[0]
            bb = nota(exa, lambda fs: [i for i, f in enumerate(fs) if f["conf"] >= 0.8], imagenes=mitad_a)[0]
            rb = [x for x in filas if x[0] == ta][0][3][0]
            ra = [x for x in filas if x[0] == tb][0][2][0]
            print(f"    cruzado: umbral {ta} (de A) en B {rb:.4f} vs base {ba:.4f} | umbral {tb} (de B) en A {ra:.4f} vs base {bb:.4f}")
