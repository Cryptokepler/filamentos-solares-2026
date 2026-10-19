"""Genera notebooks/entrenar_detector.ipynb incrustando el código de filamentos/.

Kaggle no puede leer nuestro repositorio privado, así que el notebook lleva el
código dentro (celdas %%writefile). Ejecutar desde la raíz del repositorio:
    python3 notebooks/generar_notebook_detector.py
"""
import os

import nbformat as nbf

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULOS = ["__init__.py", "datos.py", "metrica.py", "clasico.py", "observador.py", "detector.py"]

celdas = []


def md(texto):
    celdas.append(nbf.v4.new_markdown_cell(texto.strip()))


def code(texto):
    celdas.append(nbf.v4.new_code_cell(texto.strip()))


md("""
# Detector + refinador para filamentos solares

1. **Detector** (Mask R-CNN): encuentra cada filamento como un objeto separado, a 1024 px.
2. **Refinador** (U-Net pequeña): redibuja el borde de cada filamento en un recorte de la imagen original de 2048 px.
3. **Juez**: confianza mínima, unir detecciones que se pisan, tamaño mínimo y borde del Sol, elegidos en el examen fijo.

**Ronda 3:** el detector ya no estudia un número fijo de vueltas, sino hasta que la nota del examen se aplana
(meseta): en la primera meseta achica el paso de aprendizaje y en la segunda se para.

Se mide con la **PQ oficial** en el mismo examen de 121 imágenes que la ronda 1 (0,384), así que las cifras
son comparables. Incluye el informe de errores y las predicciones para el visor.
Genera `submission.csv` pero **no envía nada a la competición.**

Requisitos en Kaggle: *Accelerator* = GPU T4, *Internet* = activado (pesos preentrenados), y la competición
añadida como *Input*.
""")

code("""
# Configuración. PRUEBA_RAPIDA=True reduce todo para comprobar que el notebook funciona.
PRUEBA_RAPIDA = False

RES = 1024            # resolución del detector (la original es 2048)
PREENTRENADO = True   # Mask R-CNN preentrenado en COCO (necesita Internet en Kaggle)
# Ronda 3: el detector estudia hasta que la nota se aplana (meseta), no un número fijo de vueltas
EPOCAS_MAX = 40       # tope de seguridad
LOTE_DET = 2
LR_DET = 0.005
EXAMEN_CADA = 2       # examen cada 2 vueltas
PACIENCIA = 3         # exámenes seguidos sin mejorar para declarar meseta
MEJORA_MIN = 0.005    # lo mínimo que cuenta como mejora de PQ
ACHIQUES = 1          # en la 1.ª meseta se achica el paso (x0,1); en la siguiente se para
HORAS_MAX_DET = 6.5   # deja tiempo de sobra para juez, refinador y test dentro del límite de 12 h
EPOCAS_REF = 10
LOTE_REF = 16
LR_REF = 3e-4
SEMILLA = 42

if PRUEBA_RAPIDA:
    RES, PREENTRENADO, EPOCAS_MAX, EXAMEN_CADA, EPOCAS_REF = 256, False, 4, 1, 1
    PACIENCIA, MEJORA_MIN = 1, 1.0  # fuerza una meseta enseguida para probar achique y parada
""")

code("""
import subprocess, sys
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)
    import segmentation_models_pytorch as smp
try:
    import pycocotools
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True)
""")

md("## Código del proyecto\nCopiado del repositorio `filamentos-solares-2026`.")
code("import os\nos.makedirs('filamentos', exist_ok=True)")
for nombre in MODULOS:
    with open(os.path.join(RAIZ, "filamentos", nombre)) as f:
        fuente = f.read()
    code(f"%%writefile filamentos/{nombre}\n{fuente or '# paquete del proyecto'}")

md("## Datos y separación estudio / examen\nMisma separación que la ronda 1: las versiones de todos los expertos de una imagen van juntas.")
code("""
import glob, hashlib, json, time, random
import numpy as np, torch
from multiprocessing import Pool
from PIL import Image
from pycocotools import mask as mask_utils
from filamentos.datos import cargar_anotaciones, mascaras_de, mascara_a_rle, rle_a_mascara
from filamentos.metrica import Resultado, evaluar_imagen, resumen
from filamentos import clasico, detector as D, observador as O

random.seed(SEMILLA); np.random.seed(SEMILLA); torch.manual_seed(SEMILLA)
disp = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()

json_path = glob.glob('/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)
if not json_path:  # ejecución fuera de Kaggle
    json_path = glob.glob(os.environ.get('MAGFILO', '.') + '/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json', recursive=True)
assert json_path, 'No encuentro los datos: añade la competición como Input del notebook'
DATOS = os.path.dirname(os.path.dirname(json_path[0]))
print('Datos en', DATOS)

ANOT = cargar_anotaciones(DATOS)
nombres = sorted(n for n in ANOT if os.path.exists(f'{DATOS}/train/train_images/{n}'))

def es_val(n):
    return int(hashlib.md5(n.encode()).hexdigest(), 16) % 100 < 15
val = [n for n in nombres if es_val(n)]
tr = [n for n in nombres if not es_val(n)]
if PRUEBA_RAPIDA:
    tr, val = tr[:16], val[:6]
print(len(tr), 'imágenes de estudio,', len(val), 'de examen')
""")

code("""
def radio(img):
    \"\"\"Distancia al centro del disco solar, en radios, para cada píxel.\"\"\"
    cx, cy, R = clasico.disco_solar(img.astype(np.float32))
    yy, xx = np.indices(img.shape)
    return (np.hypot(xx - cx, yy - cy) / R).astype(np.float32)

def cargar(n):
    img = D.cargar_gris(f'{DATOS}/train/train_images/{n}', RES)
    return img, radio(img)

t = time.time()
with Pool(os.cpu_count()) as p:
    cache = dict(zip(tr + val, p.map(cargar, tr + val)))
ejemplos_tr = [(cache[n][0], anns) for n in tr for anns in ANOT[n].values()]
print(f'{len(ejemplos_tr)} ejemplos de estudio (imagen x experto), cargados en {time.time() - t:.0f} s')

# Filamentos de los expertos en el examen, a 2048 y comprimidos (RLE) para no llenar la memoria
reales_val = {n: [[D.a_rle(m) for m in mascaras_de(anns)] for anns in ANOT[n].values()] for n in val}
""")

md("## Examen con la PQ oficial")
code("""
def pq_examen(pred_por_imagen, nombres=None):
    \"\"\"pred_por_imagen: {imagen: [máscaras 2048]}. Compara contra cada experto, como en la ronda 1.\"\"\"
    tot = Resultado()
    for n in (nombres or val):
        pred = pred_por_imagen[n]
        for rles in reales_val[n]:
            tot += evaluar_imagen([mask_utils.decode(r).astype(bool) for r in rles], pred)
    return tot

def detectar_examen(modelo):
    \"\"\"Detecciones crudas del examen, comprimidas: {imagen: (rles a RES, confianzas)}.\"\"\"
    out = {}
    for n in val:
        with torch.autocast(disp, enabled=disp == 'cuda'):
            m, c = D.detectar(modelo, cache[n][0], disp)
        keep = c >= 0.2
        out[n] = ([D.a_rle(x) for x in m[keep]], c[keep])
    return out

def juez(crudas, n, **post):
    rles, conf = crudas[n]
    m = np.stack([mask_utils.decode(r).astype(bool) for r in rles]) if rles else np.zeros((0, RES, RES), bool)
    return D.instancias(m, conf, cache[n][1], **post)
""")

md("## Entrenamiento del detector")
code("""
det = D.construir_detector(PREENTRENADO, RES).to(disp)
params = [p for p in det.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=LR_DET, momentum=0.9, weight_decay=1e-4)
escalador = torch.amp.GradScaler(enabled=disp == 'cuda')
dl = torch.utils.data.DataLoader(D.DatosDetector(ejemplos_tr, RES, aumentar=True), batch_size=LOTE_DET,
                                 shuffle=True, num_workers=2, collate_fn=D.juntar)

mejor, historia = -1, []
sin_mejora, achiques, t_det = 0, 0, time.time()
for epoca in range(1, EPOCAS_MAX + 1):
    det.train(); t = time.time(); suma = 0
    for imgs, objs in dl:
        imgs = [i.to(disp) for i in imgs]
        objs = [{k: v.to(disp) for k, v in o.items()} for o in objs]
        with torch.autocast(disp, enabled=disp == 'cuda'):
            l = sum(det(imgs, objs).values())
        opt.zero_grad(); escalador.scale(l).backward(); escalador.step(opt); escalador.update()
        suma += l.item()
    linea = f'vuelta {epoca}: pérdida {suma / len(dl):.4f} ({time.time() - t:.0f} s, paso {opt.param_groups[0]["lr"]:g})'
    parar = False
    if epoca % EXAMEN_CADA == 0:
        crudas = detectar_examen(det)
        res = pq_examen({n: juez(crudas, n, umbral=0.6) for n in val})
        historia.append((epoca, round(res.pq, 4)))
        linea += ' | ' + resumen(res)
        if res.pq > mejor:
            if res.pq >= mejor + MEJORA_MIN:
                sin_mejora = 0
            else:
                sin_mejora += 1  # mejora demasiado pequeña: se guarda, pero cuenta para la meseta
            mejor = res.pq
            torch.save(det.state_dict(), 'detector_mejor.pt')
            linea += ' | guardado'
        else:
            sin_mejora += 1
        linea += f' | sin mejora {sin_mejora}/{PACIENCIA}'
        if sin_mejora >= PACIENCIA:
            if achiques < ACHIQUES:
                achiques += 1; sin_mejora = 0
                for g in opt.param_groups:
                    g['lr'] *= 0.1
                linea += ' | MESETA: achico el paso x0,1'
            else:
                linea += ' | MESETA otra vez: paro'
                parar = True
    if time.time() - t_det > HORAS_MAX_DET * 3600:
        linea += ' | tope de tiempo: paro'
        parar = True
    print(linea, flush=True)
    if parar:
        break
if not os.path.exists('detector_mejor.pt'):
    torch.save(det.state_dict(), 'detector_mejor.pt')
print('Curva (vuelta, PQ):', historia)
print(f'Entrenamiento del detector: {epoca} vueltas en {(time.time() - t_det) / 60:.0f} min')
print('Mejor PQ del detector (confianza 0,6):', round(mejor, 4))
""")

md("## Juez del detector\nConfianza mínima, si se unen las detecciones que se pisan, y tamaño mínimo.")
code("""
det.load_state_dict(torch.load('detector_mejor.pt', map_location=disp))
crudas = detectar_examen(det)
resultados = []
for umbral in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9):  # ronda 2 ganó en 0,8, el borde de la rejilla
    for fusionar in (True, False):
        for area_min in (0, 330):
            post = dict(umbral=umbral, fusionar=fusionar, area_min=area_min)
            res = pq_examen({n: juez(crudas, n, **post) for n in val})
            resultados.append((res.pq, post))
            print(post, '->', resumen(res), flush=True)
PQ_DET, MEJOR_POST = max(resultados, key=lambda t: t[0])
print('Mejor juez:', MEJOR_POST, 'PQ', round(PQ_DET, 4))
pred_det = {n: juez(crudas, n, **MEJOR_POST) for n in val}
""")

md("## Entrenamiento del refinador\nUn recorte por filamento de cada experto, de la imagen original de 2048 px.")
code("""
def recortes_de(n):
    img = np.asarray(Image.open(f'{DATOS}/train/train_images/{n}').convert('L'), np.uint8)
    return [par for anns in ANOT[n].values() for par in D.recortes_entreno(img, mascaras_de(anns), aumentar=True)]

t = time.time()
with Pool(os.cpu_count()) as p:
    pares = [par for lista in p.map(recortes_de, tr) for par in lista]
Xr = np.stack([a for a, _ in pares]); Yr = np.stack([b for _, b in pares]); del pares
print(f'{len(Xr)} recortes en {time.time() - t:.0f} s')

ref = smp.Unet('resnet18', encoder_weights='imagenet' if PREENTRENADO else None, in_channels=3, classes=1).to(disp)
opt = torch.optim.AdamW(ref.parameters(), lr=LR_REF, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS_REF)
escalador = torch.amp.GradScaler(enabled=disp == 'cuda')
dl = torch.utils.data.DataLoader(D.DatosRecortes(Xr, Yr), batch_size=LOTE_REF, shuffle=True, drop_last=True, num_workers=2)
for epoca in range(1, EPOCAS_REF + 1):
    ref.train(); t = time.time(); suma = 0
    for x, y in dl:
        x, y = x.to(disp), y.to(disp)
        with torch.autocast(disp, enabled=disp == 'cuda'):
            l = O.perdida(ref(x), y)  # BCE + Dice, la misma pérdida de la ronda 1
        opt.zero_grad(); escalador.scale(l).backward(); escalador.step(opt); escalador.update()
        suma += l.item()
    sched.step()
    print(f'época {epoca}: pérdida {suma / len(dl):.4f} ({time.time() - t:.0f} s)', flush=True)
torch.save(ref.state_dict(), 'refinador.pt')
del Xr, Yr
""")

md("## ¿Mejora el refinador?\nMismo juez del detector; se prueba el umbral del refinador.")
code("""
def gris2048(n, carpeta='train/train_images'):
    return np.asarray(Image.open(f'{DATOS}/{carpeta}/{n}').convert('L'), np.uint8)

USAR_REF, UMBRAL_REF, PQ_FINAL = False, None, PQ_DET
print('Detector solo:', resumen(pq_examen(pred_det)))
for umbral_ref in (0.3, 0.4, 0.5):  # ronda 2 ganó en 0,5, el borde de la rejilla
    pred_ref = {}
    for n in val:
        with torch.autocast(disp, enabled=disp == 'cuda'):
            pred_ref[n] = D.refinar(ref, gris2048(n), pred_det[n], disp, umbral=umbral_ref)
    res = pq_examen(pred_ref)
    print(f'Detector + refinador (umbral {umbral_ref}):', resumen(res), flush=True)
    if res.pq > PQ_FINAL:
        USAR_REF, UMBRAL_REF, PQ_FINAL, pred_final = True, umbral_ref, res.pq, pred_ref
if not USAR_REF:
    pred_final = pred_det
print(f'Elegido: {"detector + refinador (umbral " + str(UMBRAL_REF) + ")" if USAR_REF else "detector solo"} -> PQ {PQ_FINAL:.4f}')
print('Ronda 1 (U-Net + juez) para comparar: PQ 0,384')
""")

md("""
## Predicciones del examen para el visor
Guarda `predicciones_examen.json` en *Output* para compararlo en el visor con los dibujos de los expertos.
""")
code("""
from filamentos.datos import mascara_a_poligonos
examen = {}
for n in val:
    pred = pred_final[n]
    res = pq_examen(pred_final, [n])
    examen[n] = {'pq': round(res.pq, 3), 'tp': res.tp, 'fp': res.fp, 'fn': res.fn,
                 'f': [{'a': int(m.sum()), 'p': mascara_a_poligonos(m)} for m in pred]}
with open('predicciones_examen.json', 'w') as fh:
    json.dump({'post': MEJOR_POST, 'refinador': UMBRAL_REF, 'imagenes': examen}, fh, separators=(',', ':'))
peores = sorted(examen.items(), key=lambda t: t[1]['pq'])[:10]
print('Las 10 imágenes del examen donde peor lo hace:')
for n, e in peores:
    print(f"  {n}: PQ {e['pq']:.2f}  (aciertos {e['tp']}, inventados {e['fp']}, se le escapan {e['fn']})")
""")

md("""
## Informe de errores
Cada filamento inventado (FP) y cada filamento que se le escapa (FN) se clasifica por tamaño, distancia al
borde del Sol, oscuridad y telescopio. Y por tipo de error:
- **casi**: lo vio pero lo partió, lo fusionó o dibujó mal el borde (mejor IoU entre 0,1 y 0,5)
- **puro**: inventado de la nada o no visto en absoluto (mejor IoU < 0,1)
""")
code("""
import pandas as pd
from filamentos.metrica import detalle_errores
TELESCOPIO = {'B': 'Big Bear', 'M': 'Mauna Loa', 'L': 'Learmonth', 'U': 'Udaipur', 'T': 'Teide', 'C': 'Cerro Tololo'}
filas = []
for n in val:
    pred = pred_final[n]
    x, r = O.preparar_imagen(f'{DATOS}/train/train_images/{n}', 1024)
    r_g = np.asarray(Image.fromarray(r).resize((2048, 2048), Image.BILINEAR))
    contraste = np.asarray(Image.fromarray(x[1]).resize((2048, 2048), Image.BILINEAR)) * 1.5
    tel = TELESCOPIO.get(os.path.splitext(n)[0][-2], '?')
    for anot, rles in zip(ANOT[n], reales_val[n]):
        reales = [mask_utils.decode(r).astype(bool) for r in rles]
        d_r, d_p = detalle_errores(reales, pred)
        for lado, mascaras, dets in (('real', reales, d_r), ('red', pred, d_p)):
            for m, (ok, iou) in zip(mascaras, dets):
                filas.append(dict(imagen=n, anotador=anot, lado=lado, acierto=ok, iou=iou, area=int(m.sum()),
                                  radio=float(r_g[m].mean()), contraste=float(contraste[m].mean()), telescopio=tel))
E = pd.DataFrame(filas)
E['error'] = np.where(E.acierto, 'acierto', np.where(E.iou >= 0.1, 'casi', 'puro'))
E['tamaño'] = pd.cut(E.area, [0, 330, 1000, 3000, 10000, 1e9], labels=['<330', '330-1k', '1k-3k', '3k-10k', '>10k'])
E['zona'] = pd.cut(E.radio, [0, 0.5, 0.8, 0.9, 2], labels=['centro <0,5', '0,5-0,8', '0,8-0,9', 'borde >0,9'])
E['oscuridad'] = pd.cut(E.contraste, [0, 0.8, 0.9, 0.95, 9], labels=['muy oscuro <0,8', '0,8-0,9', '0,9-0,95', 'casi fondo >0,95'])
E.to_csv('errores_examen.csv', index=False)

def tabla(col):
    t = pd.crosstab([E.lado, E[col]], E.error)
    t['% acierto'] = (100 * t.get('acierto', 0) / t.sum(axis=1)).round(0)
    return t

pd.set_option('display.width', 200)
for col in ('tamaño', 'zona', 'oscuridad', 'telescopio'):
    print(f'\\n=== Por {col} (real = filamentos de los expertos; red = lo que dibujó el modelo) ===')
    print(tabla(col))
""")

md("## Predicción del test y `submission.csv`\nEl archivo se genera pero **no se envía**.")
code("""
import csv
test_dir = f'{DATOS}/test/test_images'
filas = []
for f in (sorted(os.listdir(test_dir)) if os.path.isdir(test_dir) else []):
    grande = gris2048(f, 'test/test_images')
    img = np.asarray(Image.fromarray(grande).resize((RES, RES), Image.BILINEAR), np.uint8)
    with torch.autocast(disp, enabled=disp == 'cuda'):
        m, c = D.detectar(det, img, disp)
        pred = D.instancias(m, c, radio(img), **MEJOR_POST)
        if USAR_REF:
            pred = D.refinar(ref, grande, pred, disp, umbral=UMBRAL_REF)
    img_id = os.path.splitext(f)[0]
    for k, mk in enumerate(pred, start=1):
        filas.append((f'{img_id}_{k}', mascara_a_rle(mk)))
    if PRUEBA_RAPIDA and len(filas) > 20:
        break
with open('submission.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['filament_id', 'segmentation_rle'])
    w.writerows(filas)

ids = [a for a, _ in filas]
assert len(ids) == len(set(ids))
if filas:
    assert rle_a_mascara(filas[0][1]).shape == (2048, 2048)
else:
    print('AVISO: no hay filamentos en el test (¿falta la carpeta test?)')
print(len(filas), 'filamentos en', len(set(i.rsplit('_', 1)[0] for i in ids)), 'imágenes -> submission.csv')
print(f'Tiempo total: {(time.time() - T0) / 60:.0f} min')
""")

nb = nbf.v4.new_notebook()
nb["cells"] = celdas
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "accelerator": "GPU"}
salida = os.path.join(RAIZ, "notebooks", "entrenar_detector.ipynb")
nbf.write(nb, salida)
print("Notebook generado:", salida)
