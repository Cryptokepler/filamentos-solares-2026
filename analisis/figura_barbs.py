import json,sys,numpy as np,cv2
sys.path.insert(0,'/home/claude/filamentos-solares-2026')
from analisis.barbs import nuestras, parejas, fina
from analisis.tribunal_sim import reales_rle
from filamentos.datos import cargar_anotaciones
from pycocotools import mask as M
from PIL import Image
raiz='/mnt/user-data/uploads/filamentos/MAGFiLO_1.0_Kaggle_2026'
cand=json.load(open(sys.argv[1]))
anot=cargar_anotaciones(raiz)
items=[]
for n in sorted(cand):
    ex=reales_rle(anot[n]); pred=nuestras(cand[n],0.25)
    for e in ex:
        for p,r in parejas(pred,e):
            me=M.decode(r); a=me.sum()
            if a<4000: continue
            ys,xs=np.nonzero(me|M.decode(p)); bb=(ys.min(),ys.max(),xs.min(),xs.max())
            sub=me[bb[0]:bb[1]+1,bb[2]:bb[3]+1]; f=fina(sub,4).sum()/a
            items.append((f,n,p,r,bb))
items.sort(key=lambda x:-x[0])
pick=[items[i] for i in (0,4,10,20)]
tiles=[]
for f,n,p,r,(y0,y1,x0,x1) in pick:
    img=np.asarray(Image.open(f'{raiz}/train/train_images/{n}').convert('L'))
    m=30; y0=max(y0-m,0);x0=max(x0-m,0);y1=min(y1+m,2047);x1=min(x1+m,2047)
    crop=img[y0:y1+1,x0:x1+1]
    s=400/crop.shape[0]
    big=cv2.resize(crop,(int(crop.shape[1]*s),400),interpolation=cv2.INTER_CUBIC)
    base=cv2.cvtColor(big,cv2.COLOR_GRAY2BGR); c=base.copy()
    for mk,col in ((M.decode(r),(0,220,0)),(M.decode(p),(0,0,255))):
        mm=cv2.resize(mk[y0:y1+1,x0:x1+1],(big.shape[1],400),interpolation=cv2.INTER_NEAREST)
        cs,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(c,cs,-1,col,2)
    tiles.append(np.concatenate([base,np.full((400,6,3),255,np.uint8),c],1))
W=max(t.shape[1] for t in tiles)
tiles=[np.pad(t,((0,10),(0,W-t.shape[1]),(0,0)),constant_values=255) for t in tiles]
cv2.imwrite(sys.argv[2],np.concatenate(tiles,0))
print([(round(x[0],3),x[1]) for x in pick])
