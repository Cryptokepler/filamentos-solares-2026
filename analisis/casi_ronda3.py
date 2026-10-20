"""Clasifica los "casi" de la ronda 3: partidos, fusionados y mal dibujados, frente a experto contra experto.

Uso: python3 analisis/casi_ronda3.py predicciones_examen.json (Output del notebook de la ronda 3).
Necesita MAGFiLO en R (solo las anotaciones; no hacen falta las imágenes).
"""
import json, sys, itertools, collections
import numpy as np
from pycocotools import mask as mu
sys.path.insert(0, '/home/claude/filamentos-solares-2026')
from filamentos.datos import cargar_anotaciones
R = '/mnt/user-data/uploads/filamentos/MAGFiLO_1.0_Kaggle_2026'
A = cargar_anotaciones(R)
P = json.load(open(sys.argv[1]))['imagenes']

def rles_poly(polys):
    polys = [p for p in polys if len(p) >= 6]
    return mu.merge(mu.frPyObjects(polys, 2048, 2048)) if polys else None

def clasificar(reales, preds):
    """reales, preds: listas de RLE. Devuelve tipo por real y por pred."""
    ar = mu.area(reales) if reales else np.zeros(0); ap = mu.area(preds) if preds else np.zeros(0)
    if not reales or not preds:
        return ['escapado'] * len(reales), ['inventado puro'] * len(preds), ar, ap
    iou = np.asarray(mu.iou(preds, reales, [0] * len(reales)))  # [P,R]
    # intersección = iou*(ap+ar)/(1+iou)
    inter = iou * (ap[:, None] + ar[None, :]) / (1 + iou)
    toca = inter > 0.1 * np.minimum(ap[:, None], ar[None, :])
    tr, tp_ = [], []
    for j in range(len(reales)):
        if iou[:, j].max() > 0.5: tr.append('acierto'); continue
        ps = np.flatnonzero(toca[:, j])
        if len(ps) == 0: tr.append('escapado puro'); continue
        if len(ps) >= 2: tr.append('partido'); continue
        p = ps[0]
        if toca[p].sum() >= 2: tr.append('fusionado con otro'); continue
        tr.append('mal dibujado')
    for i in range(len(preds)):
        if iou[i].max() > 0.5: tp_.append('acierto'); continue
        rs = np.flatnonzero(toca[i])
        if len(rs) == 0: tp_.append('inventado puro'); continue
        if len(rs) >= 2: tp_.append('une varios reales'); continue
        j = rs[0]
        if toca[:, j].sum() >= 2: tp_.append('trozo de uno partido'); continue
        tp_.append('mal dibujado')
    return tr, tp_, ar, ap

def tam(a):
    return '<1k' if a < 1000 else '1k-3k' if a < 3000 else '3k-10k' if a < 10000 else '>10k'

cr = collections.Counter(); cp = collections.Counter(); ce = collections.Counter()
for n, e in P.items():
    preds = [r for r in (rles_poly(f['p']) for f in e['f']) if r is not None]
    exp = list(A[n].values())
    rl = [[mu.merge(mu.frPyObjects(a['segmentation'], 2048, 2048)) for a in anns] for anns in exp]
    for reales in rl:
        tr, tp_, ar, ap = clasificar(reales, preds)
        for t, a in zip(tr, ar): cr[(tam(a), t)] += 1
        for t, a in zip(tp_, ap): cp[(tam(a), t)] += 1
    # experto contra experto (referencia: cuánto discrepan ellos)
    for x, y in itertools.permutations(range(len(rl)), 2):
        tr, _, ar, _ = clasificar(rl[x], rl[y])
        for t, a in zip(tr, ar): ce[(tam(a), t)] += 1

def tabla(c, titulo):
    tipos = sorted({t for _, t in c}); tams = ['<1k', '1k-3k', '3k-10k', '>10k']
    print('\n' + titulo); print('%-24s' % '' + ''.join('%9s' % t for t in tams) + '    total')
    for t in tipos:
        fila = [c[(s, t)] for s in tams]; print('%-24s' % t + ''.join('%9d' % v for v in fila) + '%9d' % sum(fila))
    tot = [sum(c[(s, t)] for t in tipos) for s in tams]; print('%-24s' % 'TOTAL' + ''.join('%9d' % v for v in tot) + '%9d' % sum(tot))
tabla(cr, 'FILAMENTOS REALES (de cada experto) frente a la ronda 3')
tabla(cp, 'PREDICCIONES de la ronda 3 frente a cada experto')
tabla(ce, 'EXPERTO frente a OTRO EXPERTO (mismas imágenes con 2-3 expertos)')
