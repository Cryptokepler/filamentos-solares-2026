"""Prueba sin GPU (20 oct 2026): ¿mejora la ronda 3 quitando predicciones con los criterios de
05-criterio-pequenos.md (pálido, grupo, borde, redondeado)? Uso: python3 analisis/prueba_juez_criterio.py <carpeta con r3out/>"""
import json, sys, numpy as np, pandas as pd
from pycocotools import mask as mu
sys.path.insert(0,'.')
from filamentos import detector as D
S=sys.argv[1]
d=json.load(open(f'{S}/r3out/predicciones_examen.json'))['imagenes']
e=pd.read_csv(f'{S}/r3out/errores_examen.csv')
# rasgos por predicción
rows=[]
for img,v in d.items():
    fs=v['f']
    if not fs: continue
    rl=[mu.merge(mu.frPyObjects(f['p'],2048,2048)) for f in fs]
    bb=mu.toBbox(rl); c=bb[:,:2]+bb[:,2:]/2
    for i,f in enumerate(fs):
        dd=np.hypot(*(c-c[i]).T); dd[i]=np.inf
        j=int(np.argmin(dd)) if len(fs)>1 else None
        m=mu.decode(rl[i]).astype(bool)
        rows.append(dict(imagen=img,area=f['a'],alarg=D.alargamiento(m),
            dv=dd[j] if j is not None else np.inf, av=fs[j]['a'] if j is not None else 0))
P=pd.DataFrame(rows)
red=e[e.lado=='red']
print('preds json',len(P),'filas red',len(red), 'unicas red', red.groupby(['imagen','area']).ngroups)
P.to_csv(f'{S}/rasgos_pred.csv',index=False)

e=pd.read_csv(f'{S}/r3out/errores_examen.csv'); P=pd.read_csv(f'{S}/rasgos_pred.csv')
red=e[e.lado=='red'].merge(P,on=['imagen','area'],how='left',validate='m:1')
assert red.alarg.notna().all()
real=e[e.lado=='real']
TP0=real.acierto.sum(); FN0=(~real.acierto).sum(); FP0=(~red.acierto).sum(); IOU0=red[red.acierto].iou.sum()
def pq(tp,fp,fn,s): return s/(tp+.5*fp+.5*fn)
print('base', TP0,FP0,FN0, round(pq(TP0,FP0,FN0,IOU0),4), 'SQ',round(IOU0/TP0,3))
red['grupo']=(red.dv<150)&(red.av<3000)
def quita(mask, nombre):
    q=red[mask]; tp=q.acierto.sum(); fp=(~q.acierto).sum(); s=q[q.acierto].iou.sum()
    v=pq(TP0-tp,FP0-fp,FN0+tp,IOU0-s)
    print(f'{nombre:45s} quita aciertos {tp:3d} inventos {fp:3d} -> PQ {v:.4f} ({v-pq(TP0,FP0,FN0,IOU0):+.4f})')
peq=red.area<3000
for t in [0.92,0.93,0.94,0.95,0.96]: quita(peq&(red.contraste>t), f'pequeño y pálido (>{t})')
for r in [0.8,0.85,0.9]: quita(peq&(red.radio>r), f'pequeño y radio>{r}')
quita(peq&red.grupo,'pequeño en grupo de pequeños')
quita(peq&(red.alarg<2),'pequeño y redondeado (<2)')
quita(peq&(red.contraste>0.94)&red.grupo,'pequeño, pálido>0,94 y en grupo')
quita(peq&((red.contraste>0.95)|(red.radio>0.85)),'pequeño y (pálido>0,95 o radio>0,85)')
# ¿qué fracción de aciertos/inventos por oscuridad en pequeños?
red['oc']=pd.cut(red.contraste,[0,.85,.9,.93,.95,2])
print(red[peq].groupby('oc',observed=True).acierto.agg(['mean','size']).round(2))
red.to_csv(f'{S}/red_rasgos.csv',index=False)
