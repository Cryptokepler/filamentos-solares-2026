"""Techo con dibujos perfectos en las imágenes con 2-3 expertos: copiar a un experto, la mayoría, la unión o solo los unánimes.
Uso: python3 analisis/oraculo_consenso.py <carpeta MAGFiLO_1.0_Kaggle_2026>"""
import sys, numpy as np
from pycocotools import mask as mu
sys.path.insert(0,'.')
from filamentos import datos
anot=datos.cargar_anotaciones(sys.argv[1])
def rl(a): return [mu.merge(mu.frPyObjects(x['segmentation'],2048,2048)) for x in a]
def evalua(pred, reales_por_exp, acc):
    for R in reales_por_exp:
        if pred and R:
            iou=np.asarray(mu.iou(pred,R,[0]*len(R)))
            m=iou>0.5; tp=m.any(0).sum(); s=iou[m].sum()
        else: tp=0; s=0
        acc[0]+=tp; acc[1]+=len(pred)-tp; acc[2]+=len(R)-tp; acc[3]+=s
pq=lambda a: a[3]/(a[0]+.5*a[1]+.5*a[2])
est={k:[0,0,0,0.] for k in ['copia de un experto','mayoría (≥2 expertos)','unión (cualquiera)','solo unánimes']}
for nombre,pe in anot.items():
    if len(pe)<2: continue
    exps=sorted(pe); R=[rl(pe[e]) for e in exps]
    # agrupar filamentos de todos los expertos por IoU>0.5
    todos=[(k,r) for k,Rk in enumerate(R) for r in Rk]
    grupos=[]
    for k,r in todos:
        for g in grupos:
            if k not in g['exp'] and mu.iou([r],[g['rep']],[0])[0][0]>0.5:
                g['exp'].add(k); break
        else: grupos.append(dict(rep=r,exp={k}))
    n=len(exps)
    evalua(R[0],R,est['copia de un experto'])
    evalua([g['rep'] for g in grupos if len(g['exp'])>=2],R,est['mayoría (≥2 expertos)'])
    evalua([g['rep'] for g in grupos],R,est['unión (cualquiera)'])
    evalua([g['rep'] for g in grupos if len(g['exp'])==n],R,est['solo unánimes'])
for k,a in est.items(): print(f'{k:25s} PQ {pq(a):.3f}  TP {a[0]} FP {a[1]} FN {a[2]}')
