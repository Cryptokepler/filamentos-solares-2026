"""Tribunal A+B medido en el examen: juez A (recortes), juez B (ficha) y su media.

Uso: python3 analisis/tribunal_ab.py <carpeta paso1> <carpeta juez_a>
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/analisis/", 1)[0])
from analisis.juez_a import CONF_MIN  # noqa: E402
from analisis.juez_b import aplicar, calibracion, entrenar, puntuar, receta  # noqa: E402
from analisis.tribunal_sim import RASGOS, nota  # noqa: E402


def barrido(exa, clave, mitades, base):
    filas = []
    for t in (0.25, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5):
        el = aplicar(exa, receta(t, clave=clave), True)
        filas.append((t, nota(exa, el), *(nota(exa, el, imagenes=m)[0] for m in mitades)))
    for t, r, a, b in filas:
        print(f"    umbral {t:.2f}: PQ {r[0]:.4f} TP {r[2]} FP {r[3]} FN {r[4]} | mitad A {a:.4f} B {b:.4f}")
    ta = max(filas, key=lambda x: x[2])[0]
    tb = max(filas, key=lambda x: x[3])[0]
    en_b = [x for x in filas if x[0] == ta][0][3]
    en_a = [x for x in filas if x[0] == tb][0][2]
    print(f"    cruzado: umbral {ta} (elegido en A) da en B {en_b:.4f} (base {base[1]:.4f}); "
          f"umbral {tb} (elegido en B) da en A {en_a:.4f} (base {base[0]:.4f})")


if __name__ == "__main__":
    paso1, carpeta = sys.argv[1], sys.argv[2]
    with open(f"{paso1}/tribunal_cache.pkl", "rb") as fh:
        exa, est = pickle.load(fh)
    nota_a = np.load(f"{carpeta}/nota_a_examen.npy")
    i = 0
    for img in sorted(exa):
        for f in exa[img]["filas"]:
            if f["conf"] >= CONF_MIN:
                f["nota_a"] = float(nota_a[i])
                i += 1
            else:
                f["nota_a"] = 0.0
    assert i == len(nota_a)
    puntuar(entrenar(est, RASGOS), exa, RASGOS)
    for d in exa.values():
        for f in d["filas"]:
            f["nota_ab"] = (f["nota_a"] + f["nota_b"]) / 2
            f["dif"] = abs(f["nota_a"] - f["nota_b"])

    imgs = sorted(exa)
    rng = np.random.default_rng(0)
    rng.shuffle(imgs)
    mitades = (imgs[:60], imgs[60:])
    sel8 = lambda fs: [i for i, f in enumerate(fs) if f["conf"] >= 0.8]  # noqa: E731
    base = [nota(exa, sel8, imagenes=m)[0] for m in mitades]
    print(f"Base ronda 3: PQ {nota(exa, sel8)[0]:.4f} (mitades {base[0]:.4f} / {base[1]:.4f})")
    for clave, nombre in (("nota_a", "juez A (el ojo)"), ("nota_b", "juez B (la ficha)"),
                          ("nota_ab", "tribunal: media de A y B")):
        print(f"\n=== {nombre} ===")
        print(f"  AUC en los dudosos: {calibracion(exa, clave):.3f}")
        barrido(exa, clave, mitades, base)
    dud = [f for d in exa.values() for f in d["filas"] if 0.3 <= f["conf"] < 0.95]
    print(f"\nDesacuerdos A-B de más de 40 puntos en los dudosos: {sum(f['dif'] > 0.4 for f in dud)} de {len(dud)}")
    print("correlación A-B en los dudosos: %.2f" % np.corrcoef([f["nota_a"] for f in dud], [f["nota_b"] for f in dud])[0, 1])
